from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Configuración de la base de datos
DATABASE_URL = "sqlite:///./data/app_database.db"

# Crear el directorio data si no existe
os.makedirs("data", exist_ok=True)

# Crear motor de base de datos
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Crear sesión local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base declarativa para los modelos
Base = declarative_base()

# Dependencia para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()