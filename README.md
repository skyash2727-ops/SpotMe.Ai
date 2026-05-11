# SpotMe.ai

**A High-Performance Face Recognition & Photo Retrieval System**

SpotMe.ai solves a universal problem: finding yourself in massive, unorganized event photo dumps. Instead of manually scrolling through thousands of photos from a college fest, users can simply upload a selfie. The system instantly scans the database using state-of-the-art Deep Learning (InsightFace) and Approximate Nearest Neighbor (ANN) search to retrieve every photo containing their face in milliseconds.

### Project Context & Scalability Limitations
SpotMe.ai was built as a high-utility solution for closed communities, specifically to handle the massive photo dumps generated during college fests. 

While the machine learning and database architectures are highly optimized, the current iteration relies on scanning and downloading directly from user-provided Google Drive folders. Due to the inherent privacy concerns of requiring read-access to external Drive links, this architecture is not intended for large-scale, public commercial deployment. It remains a specialized, highly effective tool for trusted networks and campus environments.

## Features

* **Automated Drive Indexing:** Paste a Google Drive folder link, and the backend concurrently downloads, processes, and indexes images in the background.
* **State-of-the-Art ML Pipeline:** Utilizes the `buffalo_l` ResNet-50 model (trained with ArcFace loss) to extract highly accurate 512-dimensional facial embeddings.
* **Lightning-Fast Retrieval:** Leverages PostgreSQL with the `pgvector` extension and HNSW (Hierarchical Navigable Small World) indexing for millisecond cosine-similarity searches across large datasets.
* **Asynchronous Backend:** Built with FastAPI and `BackgroundTasks` to keep the UI highly responsive during heavy computational loads.
* **Modern React UI:** A strictly typed, responsive frontend built with Vite and Tailwind CSS, featuring real-time scanning animations and masonry grid results.

## Tech Stack

**Machine Learning & Vision:**
* **InsightFace (`buffalo_l`):** Face detection, alignment (Affine transformations), and embedding generation.
* **OpenCV (`cv2`) & NumPy:** In-memory image decoding, resizing, and matrix transformations.
* **ONNX Runtime:** Hardware-agnostic model execution (configured for CPU Execution Provider for macOS compatibility).

**Backend & Database:**
* **FastAPI & Uvicorn:** High-performance asynchronous REST API.
* **PostgreSQL + `pgvector`:** Relational database supercharged with vector similarity search capabilities.
* **SQLAlchemy:** ORM for database management.
* **Python `ThreadPoolExecutor`:** Concurrent batch processing for Google Drive API downloads.

**Frontend:**
* **React + TypeScript:** Strict, component-based UI.
* **Vite:** Next-generation build tool configured with API proxies to bypass CORS.
* **Tailwind CSS:** Utility-first styling for responsive design.

## System Architecture

**1. The Indexing Pipeline:**
* User submits a Google Drive folder URL.
* `download.py` fetches images using concurrent threads (optimized to bypass Drive API rate limits).
* `embedding_generate.py` detects faces, aligns them, and flattens them into 512D normalized vectors.
* `database.py` saves the vectors into PostgreSQL, instantly mapping them to the HNSW graph.

**2. The Retrieval Pipeline:**
* User uploads a selfie via the React UI.
* The backend generates a 512D vector for the selfie.
* `retrieve.py` queries PostgreSQL using the `<=>` (Cosine Distance) operator against the HNSW index (threshold < 0.6).
* The frontend renders the matched photos using Google Drive's direct view endpoints.

## Future Scope

* **Multi-Face Intersections:** Allow users to search for photos containing multiple specific people simultaneously using SQL `INTERSECT` logic on the vector indexes.
* **Negative Prompts (Ghosting):** Implement `EXCEPT` clauses to filter out specific individuals from the search results.
* **Unsupervised Clustering:** Integrate algorithms like DBSCAN to automatically group and identify the "main characters" or most photographed individuals at an event without requiring an initial selfie prompt.
* **Event-Based Partitioning:** Scale the database by partitioning the PostgreSQL tables based on specific events or dates.

***

**Built by Group**
