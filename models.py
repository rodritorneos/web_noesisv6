from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base  # Asumiendo que tienes un archivo de base con la configuración de la DB


# =========================
# 🧑 Modelo Usuario
# =========================
class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

    puntajes = relationship("Puntaje", back_populates="usuario", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="usuario", cascade="all, delete-orphan")


# =========================
# 🧮 Modelo Puntaje
# =========================
class Puntaje(Base):
    __tablename__ = "puntajes"

    id = Column(Integer, primary_key=True, index=True)
    puntaje_obtenido = Column(Float)
    puntaje_total = Column(Float)
    nivel = Column(String)
    tema = Column(String, nullable=True)  # ✅ opcional

    usuario_id = Column(Integer, ForeignKey("usuarios.id"))

    usuario = relationship("Usuario", back_populates="puntajes")


# =========================
# 💬 Modelo Feedback
# =========================
class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    tema = Column(String, nullable=True)  # ✅ opcional (no rompe si no se envía)
    frase = Column(String)
    puntaje = Column(Float)
    feedback = Column(String)
    audio_url = Column(String, nullable=True)  # ✅ opcional (ruta del audio procesado)

    usuario_id = Column(Integer, ForeignKey("usuarios.id"))
    usuario = relationship("Usuario", back_populates="feedback")


# =========================
# 👨‍🏫 Modelo Docente
# =========================
class Docente(Base):
    __tablename__ = "docentes"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    institucion = Column(String)


# =========================
# 🗣️ Modelo Oracion
# =========================
class Oracion(Base):
    __tablename__ = "oraciones"

    id = Column(Integer, primary_key=True, index=True)
    tema = Column(String)
    frase = Column(String)