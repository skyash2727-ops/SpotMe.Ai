from sqlalchemy import create_engine, Column, Integer, String, text, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import database_exists, create_database
from pgvector.sqlalchemy import Vector

DATABASE_URL = "postgresql://postgres:gaira%40123@localhost:5432/images"

# 1. Database Initialization
if not database_exists(DATABASE_URL):
    create_database(DATABASE_URL)  

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 2. Schema Definition
class EventPhoto(Base):
    __tablename__ = "event_photos"
    
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=False)
    face_embedding = Column(Vector(512))

    # Moving the Index into __table_args__ is the standard SQLAlchemy practice
    __table_args__ = (
        Index(
            'hnsw_idx_cosine',
            'face_embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            # FIX: Matched the column name and switched to Cosine Distance
            postgresql_ops={'face_embedding': 'vector_cosine_ops'} 
        ),
    )

Base.metadata.create_all(bind=engine) 

# 3. Retrieval Function
def retrieve_similar_images(image_embedding: list[float]) -> list[str]:
    """
    Takes a 1D face embedding and returns a list of matching file paths.
    """
    # For Cosine Distance, lower is better. 0.6 is a standard threshold for InsightFace.
    threshold = 0.6 
    
    with SessionLocal() as session:
        # FIX: Swapped max_inner_product to cosine_distance
        results = session.query(EventPhoto).filter(
            EventPhoto.face_embedding.cosine_distance(image_embedding) < threshold
        ).order_by(
            EventPhoto.face_embedding.cosine_distance(image_embedding)
        ).all()
        
    # FIX: Extract just the file paths so your print loop works perfectly
    return [row.file_path for row in results]

def retrieve_similar_images_with_scores(
    image_embedding: list[float],
    threshold: float = 0.6,
    limit: int = 50,
) -> list[tuple[str, float]]:
    with SessionLocal() as session:
        dist = EventPhoto.face_embedding.cosine_distance(image_embedding)
        rows = (
            session.query(EventPhoto, dist.label("score"))
            .filter(dist < threshold)
            .order_by(dist)
            .limit(limit)
            .all()
        )
    return [(row.EventPhoto.file_path, row.score) for row in rows]

