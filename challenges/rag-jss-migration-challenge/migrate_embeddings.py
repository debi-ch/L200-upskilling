import os
import pg8000
import sqlalchemy
import vertexai

from google.cloud.sql.connector import Connector, IPTypes
from vertexai.language_models import TextEmbeddingModel

# --- Configuration ---
# These will be read from environment variables
INSTANCE_CONNECTION_NAME = os.environ.get("INSTANCE_CONNECTION_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION")

# --- Database Connection ---

def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """Initializes a connection pool for a Cloud SQL instance of Postgres."""
    if not all([INSTANCE_CONNECTION_NAME, DB_USER, DB_PASS, DB_NAME]):
        raise ValueError("Missing database connection environment variables.")

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
    connector = Connector()

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
            ip_type=ip_type,
        )
        return conn

    pool = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn, pool_recycle=1800)
    return pool

# --- Main Migration Logic ---

def main():
    """Connects to the database, runs migration, and updates embeddings."""
    if not all([GCP_PROJECT_ID, GCP_LOCATION]):
        raise ValueError("Missing GCP_PROJECT_ID or GCP_LOCATION environment variables.")

    print("--- Starting Embedding Migration ---")
    
    # Initialize Vertex AI
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    print(f"✅ Vertex AI initialized. Using model: text-embedding-005")

    db = connect_with_connector()
    
    with db.connect() as conn:
        print("✅ Database connection successful.")
        
        # 1. Add new embedding column if it doesn't exist
        try:
            print("1. Checking for 'embedding_005' column...")
            conn.execute(sqlalchemy.text("ALTER TABLE amenities ADD COLUMN embedding_005 VECTOR(768)"))
            conn.commit()
            print("   - Column 'embedding_005' created successfully.")
        except Exception as e:
            if "already exists" in str(e):
                print("   - Column 'embedding_005' already exists. Skipping creation.")
                conn.rollback()
            else:
                print(f"   - Error adding column: {e}")
                conn.rollback()
                return

        # 2. Fetch all amenities that haven't been migrated yet
        print("2. Fetching amenities to migrate...")
        query = sqlalchemy.text("SELECT id, name, description FROM amenities WHERE embedding_005 IS NULL")
        results = conn.execute(query).fetchall()
        
        if not results:
            print("   - No amenities to migrate. All embeddings are up to date.")
            print("--- Migration Complete ---")
            return
            
        print(f"   - Found {len(results)} amenities to migrate.")

        # 3. Generate and update embeddings in batches
        print("3. Generating and updating embeddings...")
        batch_size = 5 # Small batch size to avoid overwhelming the API for this demo
        
        for i in range(0, len(results), batch_size):
            batch = results[i:i+batch_size]
            
            # Prepare texts for embedding
            ids_in_batch = [row[0] for row in batch]
            texts_to_embed = [f"{row[1]}: {row[2]}" for row in batch]
            
            print(f"   - Processing batch {i//batch_size + 1}/{(len(results)-1)//batch_size + 1} (IDs: {ids_in_batch[0]} to {ids_in_batch[-1]})")

            # Get new embeddings
            embeddings = model.get_embeddings(texts_to_embed)
            
            # Update database
            for idx, embedding in enumerate(embeddings):
                update_query = sqlalchemy.text(
                    "UPDATE amenities SET embedding_005 = :embedding WHERE id = :id"
                )
                conn.execute(update_query, parameters={"embedding": str(embedding.values), "id": ids_in_batch[idx]})
            
            conn.commit()
            print(f"   - Batch {i//batch_size + 1} complete.")

    print("--- Migration Complete ---")


if __name__ == "__main__":
    main() 