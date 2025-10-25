import subprocess
import time
import webbrowser
import os
from threading import Thread

def run_backend():
    """Inicia el servidor FastAPI"""
    print("🚀 Iniciando Backend FastAPI...")
    subprocess.run(["uvicorn", "main:app", "--reload", "--port", "8000"])

def run_frontend():
    """Inicia el servidor HTTP para el frontend"""
    print("🌐 Iniciando Frontend...")
    subprocess.run(["python", "-m", "http.server", "8080"])

def open_browser():
    """Abre el navegador después de 3 segundos"""
    time.sleep(3)
    print("🌍 Abriendo navegador...")
    webbrowser.open("http://localhost:8080/index.html")

if __name__ == "__main__":
    print("=" * 60)
    print("🎤 ENGLISH PRONUNCIATION APP")
    print("=" * 60)
    
    # Crear threads para ejecutar backend y frontend simultáneamente
    backend_thread = Thread(target=run_backend, daemon=True)
    frontend_thread = Thread(target=run_frontend, daemon=True)
    browser_thread = Thread(target=open_browser, daemon=True)
    
    # Iniciar los servidores
    backend_thread.start()
    frontend_thread.start()
    browser_thread.start()
    
    print("\n✅ Servidores iniciados:")
    print("   📡 Backend:  http://localhost:8000")
    print("   🌐 Frontend: http://localhost:8080/index.html")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("\n⚠️  Presiona CTRL+C para detener los servidores\n")
    
    try:
        # Mantener el script corriendo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Cerrando servidores...")