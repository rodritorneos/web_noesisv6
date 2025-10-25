from typing import List
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models import Base, Feedback, Oracion, Puntaje, Usuario, Docente
from schemas import (
    UsuarioChangePassword,
    UsuarioUpdateProfile,
    UsuarioInfo,
    LoginResponse,
    UsuarioResponse,
    UsuarioRegistro,
    UsuarioLogin,
    DocenteResponse,
    DocenteRegistro,
    DocenteLogin,
    DocenteInfo,
    OracionBase,
    OracionResponse,
    FeedbackResponse,
    AnalizarPronunciacionRequest,
    AnalizarPronunciacionResponse,
    MessageResponse,
    RegistroResponse,
    EstudianteEstadisticas,
    UsuarioHistorialResponse
)
from database import get_db, engine
import requests
import re
import torch
import soundfile as sf
import numpy as np
import noisereduce as nr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from difflib import SequenceMatcher
from pydub import AudioSegment
from gtts import gTTS
import os
import shutil
from datetime import datetime
import librosa

# Crear las tablas en la base de datos
Base.metadata.create_all(bind=engine)

# Configuración
REF_FILE = "voz_ref.wav"
USER_FILE = "voz_user.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_URL = "http://localhost:11434/v1/chat/completions"
TEMAS = ["Verb to be", "Present Simple", "The verb can", "Future Perfect"]
usadas = set()

# Cargar modelo ASR con manejo de errores
try:
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english").to(DEVICE)
    print(f"✅ Modelo Wav2Vec2 cargado correctamente en {DEVICE}")
except Exception as e:
    print(f"⚠️ Error al cargar modelo Wav2Vec2: {e}")
    processor = None
    model = None

# Inicializar FastAPI
app = FastAPI()

# 📁 Crear carpeta para guardar audios accesibles
os.makedirs("audios_guardados", exist_ok=True)

# 🎧 Montar carpeta como estática (para que el frontend pueda acceder a los audios)
app.mount("/audios", StaticFiles(directory="audios_guardados"), name="audios")

@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Función para generar oraciones desde Ollama ===
def generar_oracion_tema(tema: str):
    prompt = f"Generate ONE very short and simple English sentence (A1–A2 level) for the topic '{tema}'. Example: 'He is happy.' or 'I can run fast.' Avoid repetition or complex words."
    try:
        response = requests.post(MODEL_URL, json={
            "model": "gemma3:4b-it-qat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8
        })
        response.raise_for_status()
        data = response.json()
        frase = data["choices"][0]["message"]["content"].strip()

        frase = re.sub(r'[*"`]', '', frase).strip()
        frase = frase[0].upper() + frase[1:]
        if not frase.endswith("."):
            frase += "."
        
        if frase.lower() in usadas:
            return generar_oracion_tema(tema)
        usadas.add(frase.lower())
        return frase

    except requests.RequestException as e:
        print(f"Error al generar oración con Ollama: {e}")
        return "Error en la generación de la oración."

# === Función para generar TTS ===
def generar_tts(frase: str, filename_wav: str):
    temp_mp3 = "voz_temp.mp3"
    tts = gTTS(text=frase, lang="en", slow=False)
    tts.save(temp_mp3)

    sound = AudioSegment.from_file(temp_mp3, format="mp3")
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(filename_wav, format="wav")
    os.remove(temp_mp3)

# === Función MEJORADA para procesar audio del navegador ===
def procesar_audio_navegador(audio_path: str, output_path: str):
    """
    Convierte el audio recibido del navegador a formato compatible
    con el modelo (16kHz, mono, normalizado)
    """
    try:
        # Cargar audio con librosa (más robusto que soundfile para diferentes formatos)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Reducción de ruido
        audio = nr.reduce_noise(y=audio, sr=16000, prop_decrease=0.3)
        
        # Normalizar audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Aumentar volumen ligeramente para mejor detección
        audio = audio * 1.5
        audio = np.clip(audio, -1.0, 1.0)
        
        # Guardar audio procesado
        sf.write(output_path, audio, 16000)
        return True
        
    except Exception as e:
        print(f"❌ Error al procesar audio: {e}")
        return False

# === Función MEJORADA de transcripción ===
def transcribir_audio(filename: str):
    """
    Transcribe el audio usando Wav2Vec2 con manejo de errores robusto
    """
    if model is None or processor is None:
        return ""
    
    try:
        # Leer audio
        speech, fs = sf.read(filename)
        
        # Verificar que el audio no esté vacío
        if len(speech) == 0 or np.max(np.abs(speech)) < 0.001:
            print("⚠️ Audio demasiado silencioso o vacío")
            return ""
        
        # Asegurar que sea mono
        if len(speech.shape) > 1:
            speech = np.mean(speech, axis=1)
        
        # Procesar con el modelo
        input_values = processor(speech, return_tensors="pt", sampling_rate=fs).input_values.to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        
        # Limpiar transcripción
        transcription = transcription.lower().strip()
        
        print(f"📝 Transcripción: '{transcription}'")
        return transcription
        
    except Exception as e:
        print(f"❌ Error en transcripción: {e}")
        return ""

# === Función MEJORADA para comparar textos ===
def comparar_textos(ref: str, user: str):
    """
    Compara dos textos con normalización mejorada
    """
    # Normalizar: minúsculas, quitar puntuación extra
    ref_clean = re.sub(r'[^\w\s]', '', ref.lower().strip())
    user_clean = re.sub(r'[^\w\s]', '', user.lower().strip())
    
    # Comparación básica
    similitud_basica = SequenceMatcher(None, ref_clean, user_clean).ratio()
    
    # Comparación por palabras (más tolerante a pequeños errores)
    ref_words = ref_clean.split()
    user_words = user_clean.split()
    
    if len(ref_words) == 0:
        return 0.0
    
    palabras_correctas = sum(1 for word in user_words if word in ref_words)
    similitud_palabras = palabras_correctas / len(ref_words)
    
    # Promedio ponderado (70% similitud básica, 30% palabras)
    similitud_final = (similitud_basica * 0.7) + (similitud_palabras * 0.3)
    
    return similitud_final

# === Función para identificar errores específicos ===
def identificar_errores(ref: str, user: str):
    """
    Identifica qué palabras están mal pronunciadas
    """
    ref_words = re.sub(r'[^\w\s]', '', ref.lower()).split()
    user_words = re.sub(r'[^\w\s]', '', user.lower()).split()
    
    errores = []
    for i, ref_word in enumerate(ref_words):
        if i >= len(user_words):
            errores.append(f"Falta: '{ref_word}'")
        elif ref_word != user_words[i]:
            # Verificar si es similar (por si hay pequeñas variaciones)
            if SequenceMatcher(None, ref_word, user_words[i]).ratio() < 0.6:
                errores.append(f"'{user_words[i]}' → '{ref_word}'")
    
    return errores

# =========== FUNCIONES USUARIO ========== #

def get_user_by_email(db: Session, email: str):
    return db.query(Usuario).filter(Usuario.email == email).first()

def get_user_by_username(db: Session, username: str):
    return db.query(Usuario).filter(Usuario.username == username).first()

def create_user(db: Session, username: str, email: str, password: str):
    db_user = Usuario(username=username, email=email, password=password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# =========== FUNCIONES DOCENTE ========== #
def get_teacher_by_email(db: Session, email: str):
    return db.query(Docente).filter(Docente.email == email).first()

def get_teacher_by_username(db: Session, username: str):
    return db.query(Docente).filter(Docente.username == username).first()

def create_teacher(db: Session, username: str, email: str, password: str):
    db_teacher = Docente(username=username, email=email, password=password, institucion="CCJCALLAO")
    db.add(db_teacher)
    db.commit()
    db.refresh(db_teacher)
    return db_teacher

# =========== FUNCIONES para obtener información de los estudiantes ========== #
def get_estudiante_by_id(db: Session, user_id: int):
    return db.query(Usuario).filter(Usuario.id == user_id).first()

def get_historial_evaluaciones(db: Session, user_id: int):
    return db.query(Feedback).filter(Feedback.usuario_id == user_id).all()

def get_estadisticas_progreso(db: Session, user_id: int):
    evaluaciones = get_historial_evaluaciones(db, user_id)
    if not evaluaciones:
        return {"promedio": 0, "mejor_puntaje": 0, "peor_puntaje": 0}

    puntuaciones = [eval.puntaje for eval in evaluaciones]
    promedio = sum(puntuaciones) / len(puntuaciones)
    mejor_puntaje = max(puntuaciones)
    peor_puntaje = min(puntuaciones)

    return {"promedio": promedio, "mejor_puntaje": mejor_puntaje, "peor_puntaje": peor_puntaje}

def get_estudiantes_ordenados(db: Session):
    estudiantes = db.query(Usuario).all()
    estudiantes_ordenados = []

    for estudiante in estudiantes:
        estadisticas = get_estadisticas_progreso(db, estudiante.id)
        estudiantes_ordenados.append({
            "id": estudiante.id,
            "username": estudiante.username,
            "email": estudiante.email,
            "puntaje_promedio": estadisticas["promedio"]
        })

    estudiantes_ordenados.sort(key=lambda x: x["puntaje_promedio"], reverse=True)
    return estudiantes_ordenados

# === FastAPI Endpoints ===

@app.get("/generar_oracion/{tema}", response_model=OracionBase)
def obtener_oracion(tema: str, db: Session = Depends(get_db)):
    if tema not in TEMAS:
        raise HTTPException(status_code=404, detail="Tema no encontrado")

    # Función interna para generar frase única
    def generar_frase_unica():
        frase = generar_oracion_tema(tema)

        # Revisar si ya existe en la BD
        existe = db.query(Oracion).filter(Oracion.frase.ilike(frase)).first()
        if existe:
            return generar_frase_unica()  # Generar otra si ya existe
        return frase

    # 1. Generar frase única
    frase = generar_frase_unica()

    # 2. Guardar en la tabla Oracion
    nueva_oracion = Oracion(tema=tema, frase=frase)
    db.add(nueva_oracion)
    db.commit()
    db.refresh(nueva_oracion)

    print(f"✅ Oración generada y guardada: '{frase}'")
    
    # 3. Devolver al frontend
    return nueva_oracion

# === ENDPOINT PRINCIPAL CORREGIDO ===
@app.post("/analizar_pronunciacion_audio/")
async def analizar_pronunciacion_audio(
    audio: UploadFile = File(...),
    frase: str = Form(...),
    usuario_email: str = Form(...),
    tema: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Analiza la pronunciación del usuario, genera feedback,
    guarda en Feedback y Puntaje, y asocia el audio procesado.
    """
    audio_filename = None
    audio_procesado = None

    try:
        print(f"\n{'='*60}")
        print(f"🎤 Analizando pronunciación de: {usuario_email}")
        print(f"📝 Frase objetivo: '{frase}'")

        # 1️⃣ Guardar archivo de audio temporal
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = f"temp_audio_{timestamp}.wav"
        audio_procesado = f"temp_processed_{timestamp}.wav"

        content = await audio.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="El archivo de audio está vacío")

        with open(audio_filename, "wb") as f:
            f.write(content)
        print(f"✅ Audio guardado: {audio_filename} ({len(content)} bytes)")

        # 2️⃣ Procesar audio (convertir, reducir ruido, normalizar)
        if not procesar_audio_navegador(audio_filename, audio_procesado):
            raise HTTPException(status_code=400, detail="Error al procesar el audio")
        print(f"✅ Audio procesado: {audio_procesado}")

        # 3️⃣ Transcribir el audio
        transcripcion = transcribir_audio(audio_procesado)

        if not transcripcion or len(transcripcion.strip()) == 0:
            print("⚠️ No se pudo transcribir el audio")
            transcripcion = "No se pudo transcribir"
            porcentaje = 0.0
            feedback_text = "No se detectó audio claro. Intenta hablar más cerca del micrófono 🎤"
            errores = []
        else:
            # 4️⃣ Comparar textos y calcular similitud
            similitud = comparar_textos(frase, transcripcion)
            porcentaje = round(similitud * 100, 2)

            # 5️⃣ Identificar errores
            errores = identificar_errores(frase, transcripcion)

            # 6️⃣ Generar feedback textual
            if porcentaje >= 90:
                feedback_text = "¡Excelente pronunciación! 🎉 Tu pronunciación es casi perfecta."
            elif porcentaje >= 75:
                feedback_text = "¡Muy bien! 👍 "
                if errores:
                    feedback_text += f"Revisa: {', '.join(errores[:2])}"
            elif porcentaje >= 50:
                feedback_text = "Buen intento 💪 "
                if errores:
                    feedback_text += f"Trabaja en: {', '.join(errores[:2])}"
            else:
                feedback_text = "Sigue practicando 🗣️ "
                if errores:
                    feedback_text += f"Presta atención a: {', '.join(errores[:2])}"

        # 7️⃣ Determinar nivel
        if porcentaje >= 85:
            nivel = "Avanzado"
        elif porcentaje >= 60:
            nivel = "Intermedio"
        else:
            nivel = "Básico"

        print(f"📊 Similitud: {porcentaje}%, Nivel: {nivel}")
        print(f"💬 Feedback: {feedback_text}")

        # 8️⃣ Obtener usuario
        usuario = get_user_by_email(db, usuario_email)
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        # ✅ 8.5 Crear la URL del audio procesado antes de guardar
        audio_url = None
        if os.path.exists(audio_procesado):
            audio_url = f"/audios/{os.path.basename(audio_procesado)}"

        # 9️⃣ Guardar en Feedback
        nuevo_feedback = Feedback(
            tema=tema,
            frase=frase,
            puntaje=porcentaje,
            feedback=feedback_text,
            audio_url=audio_url,
            usuario_id=usuario.id
        )
        db.add(nuevo_feedback)

        # 🔟 Guardar en Puntaje
        nuevo_puntaje = Puntaje(
            puntaje_obtenido=porcentaje,
            puntaje_total=100,
            nivel=nivel,
            usuario_id=usuario.id
        )
        db.add(nuevo_puntaje)

        db.commit()
        db.refresh(nuevo_feedback)
        db.refresh(nuevo_puntaje)
        print("✅ Feedback y puntaje guardados en la base de datos")

        # 🔁 Devolver respuesta al frontend (no rompe nada)
        return {
            "transcripcion": transcripcion,
            "porcentaje": porcentaje,
            "nivel": nivel,
            "feedback": feedback_text,
            "guardado": True,
            "audio_url": audio_url  # 🔥 opcional, no genera error si frontend no lo usa
        }

    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al procesar audio: {str(e)}")

    finally:
        # ♻️ Guardar audio procesado y limpiar temporales
        if audio_procesado and os.path.exists(audio_procesado):
            try:
                destino = os.path.join("audios_guardados", os.path.basename(audio_procesado))
                shutil.move(audio_procesado, destino)
                print(f"🎧 Audio disponible en: /audios/{os.path.basename(audio_procesado)}")
            except Exception as e:
                print(f"⚠️ No se pudo mover el audio procesado: {e}")

        for archivo in [audio_filename, REF_FILE]:
            if archivo and os.path.exists(archivo):
                try:
                    os.remove(archivo)
                except:
                    pass

# === Resto de endpoints (sin cambios) ===

@app.on_event("shutdown")
def limpiar_audios():
    for archivo in [REF_FILE, USER_FILE, "voz_temp.mp3"]:
        if os.path.exists(archivo):
            os.remove(archivo)

@app.get("/usuarios", response_model=List[UsuarioResponse])
async def get_usuarios(db: Session = Depends(get_db)):
    usuarios = db.query(Usuario).all()
    return [{"username": user.username, "email": user.email, "password": user.password} for user in usuarios]

@app.post("/usuarios/registro", response_model=RegistroResponse)
async def registrar_usuario(usuario: UsuarioRegistro, db: Session = Depends(get_db)):
    if get_user_by_email(db, usuario.email):
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    if get_user_by_username(db, usuario.username):
        raise HTTPException(status_code=400, detail="El username ya está registrado")
    
    nuevo_usuario = create_user(db, usuario.username, usuario.email, usuario.password)
    
    return {"message": "Usuario registrado exitosamente", "email": usuario.email}

@app.post("/usuarios/login", response_model=LoginResponse)
async def login_usuario(usuario: UsuarioLogin, db: Session = Depends(get_db)):
    usuario_encontrado = get_user_by_email(db, usuario.email)
    
    if not usuario_encontrado:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    if usuario_encontrado.password != usuario.password:
        raise HTTPException(status_code=401, detail="Contraseña incorrecta")
    
    return {"message": "Login exitoso", "email": usuario.email}

@app.get("/usuarios/{email}", response_model=UsuarioInfo)
async def obtener_usuario(email: str, db: Session = Depends(get_db)):
    usuario = get_user_by_email(db, email)
    
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    return {"username": usuario.username, "email": usuario.email}

@app.delete("/usuarios/{email}", response_model=MessageResponse)
async def eliminar_usuario(email: str, db: Session = Depends(get_db)):
    usuario = get_user_by_email(db, email)
    
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    db.delete(usuario)
    db.commit()
    
    return {"message": "Usuario, favoritos, visitas y puntajes eliminados exitosamente"}

@app.put("/usuarios/{email}/perfil", response_model=MessageResponse)
async def actualizar_perfil_usuario(
    email: str, 
    perfil_actualizado: UsuarioUpdateProfile, 
    db: Session = Depends(get_db)
):
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    if perfil_actualizado.username != usuario.username:
        usuario_existente = get_user_by_username(db, perfil_actualizado.username)
        if usuario_existente:
            raise HTTPException(status_code=400, detail="El nuevo username ya está registrado")
    
    usuario.username = perfil_actualizado.username
    
    db.commit()
    
    return {"message": "Perfil actualizado exitosamente"}

@app.put("/usuarios/{email}/password", response_model=MessageResponse)
async def cambiar_contraseña_usuario(
    email: str, 
    password_data: UsuarioChangePassword, 
    db: Session = Depends(get_db)
):
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    usuario.password = password_data.new_password
    db.commit()
    
    return {"message": "Contraseña actualizada correctamente"}

# === Endpoints de DOCENTE ===

@app.get("/docentes", response_model=List[DocenteResponse])
async def get_docentes(db: Session = Depends(get_db)):
    docentes = db.query(Docente).all()
    return [{"username": d.username, "email": d.email, "password": d.password, "institucion": d.institucion} for d in docentes]

@app.post("/docentes/registro", response_model=RegistroResponse)
async def registrar_docente(docente: DocenteRegistro, db: Session = Depends(get_db)):
    if get_teacher_by_email(db, docente.email):
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    if get_teacher_by_username(db, docente.username):
        raise HTTPException(status_code=400, detail="El username ya está registrado")
    
    if not docente.email.endswith("@ccjcallao.edu.pe"):
        raise HTTPException(status_code=400, detail="Debe usar un correo institucional @ccjcallao.edu.pe")
    
    nuevo_docente = create_teacher(db, docente.username, docente.email, docente.password)
    
    return {"message": "Docente registrado exitosamente", "email": docente.email}

@app.post("/docentes/login", response_model=LoginResponse)
async def login_docente(docente: DocenteLogin, db: Session = Depends(get_db)):
    docente_encontrado = get_teacher_by_email(db, docente.email)
    
    if not docente_encontrado:
        raise HTTPException(status_code=404, detail="Docente no encontrado")
    
    if docente_encontrado.password != docente.password:
        raise HTTPException(status_code=401, detail="Contraseña incorrecta")
    
    return {"message": "Login exitoso", "email": docente.email}

@app.get("/docentes/{email}", response_model=DocenteInfo)
async def obtener_docente(email: str, db: Session = Depends(get_db)):
    docente = get_teacher_by_email(db, email)
    
    if not docente:
        raise HTTPException(status_code=404, detail="Docente no encontrado")
    
    return {
        "username": docente.username,
        "email": docente.email,
        "institucion": docente.institucion
    }

@app.get("/docentes/{id}/buscar-estudiante", response_model=List[UsuarioResponse])
async def buscar_estudiante(id: int, query: str, db: Session = Depends(get_db)):
    estudiantes = db.query(Usuario).filter(
        Usuario.username.ilike(f"%{query}%") | Usuario.email.ilike(f"%{query}%")
    ).all()

    return [{"id": estudiante.id, "username": estudiante.username, "email": estudiante.email} for estudiante in estudiantes]

@app.get("/docentes/{id}/estudiante/{user_id}/detalles", response_model=UsuarioHistorialResponse)
async def obtener_detalle_estudiante(id: int, user_id: int, db: Session = Depends(get_db)):
    estudiante = get_estudiante_by_id(db, user_id)
    if not estudiante:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")

    historial_evaluaciones = get_historial_evaluaciones(db, user_id)
    estadisticas_progreso = get_estadisticas_progreso(db, user_id)

    return {
        "username": estudiante.username,
        "email": estudiante.email,
        "historial_evaluaciones": [
    {
        "tema": eval.tema,
        "frase": eval.frase,
        "puntaje": eval.puntaje,
        "feedback": eval.feedback,
        "audio_url": eval.audio_url 
    }
    for eval in historial_evaluaciones
],
        "estadisticas_progreso": estadisticas_progreso
    }

@app.get("/docentes/{id}/estudiante/{user_id}/estadisticas", response_model=EstudianteEstadisticas)
async def obtener_estadisticas_progreso(id: int, user_id: int, db: Session = Depends(get_db)):
    estudiante = get_estudiante_by_id(db, user_id)
    if not estudiante:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")

    estadisticas = get_estadisticas_progreso(db, user_id)
    return estadisticas

@app.get("/docentes/{id}/estudiantes")
async def obtener_estudiantes_ordenados(id: int, db: Session = Depends(get_db)):
    estudiantes_ordenados = get_estudiantes_ordenados(db)
    return estudiantes_ordenados

@app.get("/docentes/{id}/top-5-estudiantes", response_model=List[UsuarioResponse])
async def obtener_top_5_estudiantes(id: int, db: Session = Depends(get_db)):
    estudiantes_ordenados = get_estudiantes_ordenados(db)
    return estudiantes_ordenados[:5]

@app.get("/usuarios/{email}/historial", response_model=List[FeedbackResponse])
async def obtener_historial_usuario(email: str, db: Session = Depends(get_db)):
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    historial = db.query(Feedback).filter(Feedback.usuario_id == usuario.id).all()
    return [
        {
            "tema": feedback.tema,
            "frase": feedback.frase,
            "puntaje": feedback.puntaje,
            "feedback": feedback.feedback,
            "audio_url": feedback.audio_url
        }
        for feedback in historial
    ]

@app.get("/usuarios/{email}/estadisticas")
async def obtener_estadisticas_usuario(email: str, db: Session = Depends(get_db)):
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    estadisticas = get_estadisticas_progreso(db, usuario.id)
    return estadisticas