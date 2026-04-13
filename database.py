from sqlalchemy import create_engine, Column, Integer, String, text, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import database_exists, create_database
from pgvector.sqlalchemy import Vector

# --- 1. DATABASE INITIALIZATION ---
DATABASE_URL = "postgresql://postgres:gaira%40123@localhost:5432/images"

if not database_exists(DATABASE_URL):
    create_database(DATABASE_URL)  

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# --- 2. SCHEMA DEFINITION ---
class EventPhoto(Base):
    __tablename__ = "event_photos"
    
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=False)
    face_embedding = Column(Vector(512))

    # Standard SQLAlchemy practice for adding PostgreSQL specific indexes
    __table_args__ = (
        Index(
            'hnsw_idx_cosine',
            'face_embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'face_embedding': 'vector_cosine_ops'} 
        ),
    )

# Create the tables if they don't exist yet
Base.metadata.create_all(bind=engine) 


# --- 3. DATABASE FUNCTIONS ---

def save_embedding_to_db(file_path: str, face_vector: list):
    """
    Saves a new photo and its extracted face embedding to PostgreSQL.
    """
    db = SessionLocal()
    try:
        new_photo = EventPhoto(
            file_path=file_path,
            face_embedding=face_vector 
        )
        db.add(new_photo)
        db.commit()
        # Cleaned up the print statement for a cleaner terminal output
        print(f"Successfully saved embedding for ID: {file_path}")
    except Exception as e:
        db.rollback()
        print(f"Error saving to database: {e}")
    finally:
        db.close()


def retrieve_similar_images(image_embedding: list[float]) -> list[str]:
    """
    Takes a 1D face embedding and returns a list of matching file paths.
    """
    # For Cosine Distance, lower is better. 0.6 is a standard threshold for InsightFace.
    threshold = 0.6 
    
    with SessionLocal() as session:
        results = session.query(EventPhoto).filter(
            EventPhoto.face_embedding.cosine_distance(image_embedding) < threshold
        ).order_by(
            EventPhoto.face_embedding.cosine_distance(image_embedding)
        ).all()
        
    return [row.file_path for row in results]