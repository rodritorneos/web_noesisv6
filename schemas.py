from pydantic import BaseModel, EmailStr
from typing import List, Optional  # 👈 necesario para campos opcionales


# === Modelo de Login Respuesta (Para Docentes y Usuarios) ===
class LoginResponse(BaseModel):
    message: str
    email: EmailStr

    class Config:
        orm_mode = True


# === Modelos de Usuario ===
class UsuarioResponse(BaseModel):
    username: str
    email: EmailStr
    password: str

    class Config:
        orm_mode = True


class UsuarioRegistro(BaseModel):
    username: str
    email: EmailStr
    password: str


class UsuarioLogin(BaseModel):
    email: EmailStr
    password: str


class UsuarioUpdateProfile(BaseModel):
    username: str


class UsuarioChangePassword(BaseModel):
    new_password: str


class UsuarioInfo(BaseModel):
    username: str
    email: EmailStr

    class Config:
        orm_mode = True


# === Modelos de Docente ===
class DocenteResponse(BaseModel):
    username: str
    email: EmailStr
    password: str
    institucion: str

    class Config:
        orm_mode = True


class DocenteRegistro(BaseModel):
    username: str
    email: EmailStr
    password: str


class DocenteLogin(BaseModel):
    email: EmailStr
    password: str


class DocenteInfo(BaseModel):
    username: str
    email: EmailStr
    institucion: str

    class Config:
        orm_mode = True


# === Otros Modelos ===
class OracionBase(BaseModel):
    tema: str
    frase: str

    class Config:
        orm_mode = True


class OracionResponse(BaseModel):
    tema: str
    frase: str

    class Config:
        orm_mode = True


# 💬 Feedback (con campos opcionales para compatibilidad total)
class FeedbackResponse(BaseModel):
    tema: Optional[str] = None  # 👈 opcional para no romper si no se envía
    frase: str
    puntaje: float
    feedback: str
    audio_url: Optional[str] = None  # 👈 nuevo campo opcional

    class Config:
        orm_mode = True


# 🎧 Respuesta del análisis de pronunciación
class AnalizarPronunciacionRequest(BaseModel):
    frase: str


class AnalizarPronunciacionResponse(BaseModel):
    transcripcion: str
    porcentaje: float
    feedback: str
    nivel: Optional[str] = None  # 👈 opcional
    guardado: Optional[bool] = None  # 👈 opcional
    audio_url: Optional[str] = None  # 👈 opcional
    tema: Optional[str] = None  # 👈 agregado también por si deseas incluirlo

    class Config:
        orm_mode = True


# === Mensajes y respuestas generales ===
class MessageResponse(BaseModel):
    message: str


class RegistroResponse(BaseModel):
    message: str
    email: EmailStr


# 📊 Estadísticas y progreso
class EstudianteEstadisticas(BaseModel):
    promedio: float
    mejor_puntaje: float
    peor_puntaje: float


class UsuarioHistorialResponse(BaseModel):
    username: str
    email: EmailStr
    historial_evaluaciones: List[FeedbackResponse]
    estadisticas_progreso: EstudianteEstadisticas

    class Config:
        orm_mode = True