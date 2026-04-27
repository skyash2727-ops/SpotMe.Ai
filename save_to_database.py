from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import database_exists, create_database
from pgvector.sqlalchemy import Vector
DATABASE_URL = "postgresql://postgres:gaira%40123@localhost:5432/images"
if not database_exists(DATABASE_URL):
    create_database(DATABASE_URL)    
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
class EventPhoto(Base):
    __tablename__ = "event_photos"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=False)
   # event_name = Column(String, nullable=False) 
    face_embedding = Column(Vector(512))
Base.metadata.create_all(bind=engine)
def save_embedding_to_db(file_path: str, face_vector: list):
    db = SessionLocal()
    try:
        new_photo = EventPhoto(
            file_path=file_path,
           # event_name=event_name, 
            face_embedding=face_vector 
        )
        db.add(new_photo)
        db.commit()
        print("Succesfully Added the image to Database!")
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    finally:
        db.close()