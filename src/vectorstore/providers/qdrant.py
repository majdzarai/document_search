"""
Qdrant Vector Store Provider
=============================

WHAT IS QDRANT?
---------------
Qdrant (pronounced "quadrant") is an open-source vector database.
It's designed specifically for storing and searching embeddings efficiently.

Key features of Qdrant:
1. FAST: Uses optimized algorithms for similarity search
2. SCALABLE: Can handle millions of vectors
3. FLEXIBLE: Supports filtering, metadata, and multiple collections
4. OPEN SOURCE: Free to use, can run locally or in cloud

WHY QDRANT FOR LEARNING?
------------------------
We chose Qdrant for Step 3 because:
1. It can run IN-MEMORY (no database installation needed!)
2. The API is simple and beginner-friendly
3. It's used in production by many companies
4. Great documentation and community

HOW QDRANT STORES DATA:
-----------------------
Qdrant organizes data into "collections" (like tables in SQL databases).
Each collection stores "points" (vectors with metadata).

A "point" in Qdrant contains:
1. id: A unique identifier (number or string)
2. vector: The embedding (list of floats)
3. payload: Metadata as a dictionary (any JSON-serializable data)

Example point:
{
    "id": "doc_001",
    "vector": [0.1, 0.2, 0.3, ...],  # 1536 numbers
    "payload": {
        "text": "Original document text here",
        "source": "manual.pdf",
        "page": 5
    }
}

RUNNING MODES:
--------------
Qdrant can run in different modes:

1. IN-MEMORY (what we use here):
   - Data stored in RAM only
   - Data is LOST when the app stops
   - Perfect for learning and testing
   - No installation needed!

2. LOCAL PERSISTENT:
   - Data saved to disk
   - Data survives restarts
   - Good for development

3. CLOUD/SERVER:
   - Full Qdrant server running
   - Production-ready
   - Supports clustering and replication

For Step 3, we use IN-MEMORY mode. This is intentional because:
- You don't need to install Qdrant separately
- It's perfect for learning how vector databases work
- Later steps can upgrade to persistent storage

SIMILARITY METRICS:
-------------------
Qdrant supports different ways to measure similarity:

1. COSINE: Measures angle between vectors (most common for text)
   - Range: -1 to 1 (we use 0 to 1 for normalized vectors)
   - Best for: Text embeddings, semantic search

2. DOT: Dot product of vectors
   - Range: Any number (depends on vector magnitude)
   - Best for: When vector length matters

3. EUCLID: Euclidean distance (straight-line distance)
   - Range: 0 to infinity (lower = more similar)
   - Best for: Image embeddings, spatial data

We use COSINE because it's the standard for text embeddings.
"""

from typing import List, Dict, Any, Optional
import uuid  # For generating unique point IDs that Qdrant accepts

# Import the base class that defines our interface
from src.vectorstore.base import VectorStoreProvider

# =============================================================================
# QDRANT CLIENT IMPORT
# =============================================================================
# The qdrant-client library provides tools to interact with Qdrant.
# If you haven't installed it yet, run: pip install qdrant-client
#
# The import is wrapped in a try/except to provide a helpful error message
# if the library isn't installed.

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams,          # Configures vector settings (dimension, metric)
        Distance,              # Similarity metric enum (COSINE, DOT, EUCLID)
        PointStruct,           # Data structure for a single point (vector + metadata)
        Filter,                # For filtering search results
        FieldCondition,        # Condition for a single field
        MatchValue,            # Match a specific value
    )
    QDRANT_AVAILABLE = True
except ImportError:
    # If qdrant-client isn't installed, we'll raise a helpful error later
    QDRANT_AVAILABLE = False


class QdrantVectorStore(VectorStoreProvider):
    """
    Qdrant implementation of the VectorStoreProvider interface.

    WHAT DOES THIS CLASS DO?
    ------------------------
    This class connects to Qdrant and provides methods to:
    1. Store document embeddings (upsert)
    2. Search for similar documents (search)
    3. Delete documents (delete)
    4. Get collection statistics (count, get_info)

    HOW TO USE THIS CLASS:
    ----------------------
        # Create an in-memory Qdrant store
        store = QdrantVectorStore(
            collection_name="my_documents",
            vector_dimension=1536  # Must match your embedding model!
        )

        # Store some documents
        store.upsert(
            ids=["doc1", "doc2"],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            texts=["First doc", "Second doc"],
            metadata=[{"source": "file1.pdf"}, {"source": "file2.pdf"}]
        )

        # Search for similar documents
        results = store.search(query_embedding=[0.1, 0.2, ...], top_k=3)

    IMPORTANT: VECTOR DIMENSION
    ---------------------------
    The vector_dimension MUST match your embedding model:
    - OpenAI text-embedding-3-small: 1536
    - OpenAI text-embedding-3-large: 3072
    - Cohere embed-v3: 1024

    If dimensions don't match, you'll get errors when upserting!
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        vector_dimension: int = 1536,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Qdrant vector store.

        WHAT HAPPENS DURING INITIALIZATION:
        -----------------------------------
        1. Check if qdrant-client is installed
        2. Create a connection to Qdrant (in-memory or server)
        3. Create the collection if it doesn't exist

        CONNECTION MODES:
        -----------------
        - If NO host/port/url given → IN-MEMORY mode (data lost on restart)
        - If host + port given → Connect to local Qdrant server
        - If url given → Connect to Qdrant Cloud

        Args:
            collection_name: Name for the vector collection.
                            Like a "table name" in SQL databases.
                            Default: "rag_documents"

            vector_dimension: The size of embedding vectors.
                             MUST match your embedding model!
                             Default: 1536 (OpenAI text-embedding-3-small)

            host: Qdrant server host (optional).
                  Example: "localhost"

            port: Qdrant server port (optional).
                  Default Qdrant port: 6333

            url: Full URL for Qdrant Cloud (optional).
                 Example: "https://xyz-abc.us-east-1-0.aws.cloud.qdrant.io"

            api_key: API key for Qdrant Cloud authentication (optional).

        Raises:
            ImportError: If qdrant-client is not installed.

        Example - In-memory (simplest, for learning):
            store = QdrantVectorStore(
                collection_name="my_docs",
                vector_dimension=1536
            )

        Example - Local server:
            store = QdrantVectorStore(
                collection_name="my_docs",
                vector_dimension=1536,
                host="localhost",
                port=6333
            )

        Example - Qdrant Cloud:
            store = QdrantVectorStore(
                collection_name="my_docs",
                vector_dimension=1536,
                url="https://your-cluster.cloud.qdrant.io",
                api_key="your-api-key"
            )
        """
        # =====================================================================
        # STEP 1: Check if qdrant-client is installed
        # =====================================================================
        # We need the qdrant-client library to work with Qdrant.
        # If it's not installed, we provide a helpful error message.

        if not QDRANT_AVAILABLE:
            raise ImportError(
                "The 'qdrant-client' library is required for QdrantVectorStore.\n"
                "Please install it with: pip install qdrant-client\n"
                "Or uncomment it in requirements.txt and run: pip install -r requirements.txt"
            )

        # =====================================================================
        # STEP 2: Store configuration
        # =====================================================================
        # We save these values so we can use them later (for logging, etc.)

        self._collection_name = collection_name
        self._vector_dimension = vector_dimension

        print(f"[QdrantVectorStore] Initializing...")
        print(f"[QdrantVectorStore] Collection name: {collection_name}")
        print(f"[QdrantVectorStore] Vector dimension: {vector_dimension}")

        # =====================================================================
        # STEP 3: Create the Qdrant client (connection to Qdrant)
        # =====================================================================
        # The QdrantClient handles all communication with Qdrant.
        # We decide which mode to use based on the provided parameters.

        if url:
            # QDRANT CLOUD MODE
            # Connect to a remote Qdrant Cloud instance
            print(f"[QdrantVectorStore] Connecting to Qdrant Cloud: {url}")
            self._client = QdrantClient(url=url, api_key=api_key)

        elif host:
            # LOCAL SERVER MODE
            # Connect to a Qdrant server running locally
            actual_port = port or 6333  # Default Qdrant port
            print(f"[QdrantVectorStore] Connecting to Qdrant server: {host}:{actual_port}")
            self._client = QdrantClient(host=host, port=actual_port)

        else:
            # IN-MEMORY MODE
            # Create an in-memory database (perfect for learning!)
            # Data is stored in RAM and will be lost when the app stops
            print("[QdrantVectorStore] Using IN-MEMORY mode (data not persisted)")
            print("[QdrantVectorStore] Note: Data will be lost when the app stops")
            self._client = QdrantClient(":memory:")

        # =====================================================================
        # STEP 4: Create the collection (if it doesn't exist)
        # =====================================================================
        # A "collection" in Qdrant is like a "table" in SQL databases.
        # It holds all our vectors and has specific settings (dimension, metric).

        self._create_collection_if_not_exists()

        print(f"[QdrantVectorStore] Ready to use!")

    def _create_collection_if_not_exists(self) -> None:
        """
        Create the collection if it doesn't already exist.

        WHY A SEPARATE METHOD?
        ----------------------
        We separate this logic for clarity. The __init__ method does setup,
        this method handles the specific task of collection creation.

        WHAT IS A COLLECTION?
        ---------------------
        A collection is like a table in a database. It has:
        - A name (to identify it)
        - Vector configuration (dimension, similarity metric)
        - The actual stored vectors and their metadata

        IDEMPOTENT OPERATION:
        ---------------------
        This method is "idempotent" - calling it multiple times has the
        same effect as calling it once. If the collection exists, we
        don't create it again.
        """
        # Get list of existing collections
        existing_collections = self._client.get_collections().collections
        existing_names = [col.name for col in existing_collections]

        # Check if our collection already exists
        if self._collection_name in existing_names:
            print(f"[QdrantVectorStore] Collection '{self._collection_name}' already exists")
            return

        # Create the collection with our configuration
        print(f"[QdrantVectorStore] Creating new collection '{self._collection_name}'")

        # VectorParams configures how vectors are stored and searched
        # - size: The dimension of the vectors (e.g., 1536)
        # - distance: The similarity metric (COSINE for text embeddings)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._vector_dimension,
                distance=Distance.COSINE  # Best for text embeddings
            )
        )

        print(f"[QdrantVectorStore] Collection created successfully!")

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Insert or update vectors in the Qdrant collection.

        WHAT THIS METHOD DOES:
        ----------------------
        1. Validates the input data
        2. Converts data to Qdrant's PointStruct format
        3. Sends the points to Qdrant for storage

        HOW QDRANT STORES DATA:
        -----------------------
        Each document becomes a "point" with:
        - id: The document's unique identifier
        - vector: The embedding (list of numbers)
        - payload: The text and metadata (stored as a dictionary)

        Args:
            ids: Unique identifiers for each document.
            embeddings: The embedding vectors (from your embedding model).
            texts: The original text of each document.
            metadata: Optional additional data for each document.

        Returns:
            True if successful, False if something went wrong.

        IMPORTANT NOTES:
        ----------------
        - All lists must have the same length!
        - Embedding dimensions must match the collection's configured dimension
        - IDs must be unique within the collection
        """
        # =====================================================================
        # STEP 1: Validate input data
        # =====================================================================
        # Make sure all lists have the same length and data is valid

        if len(ids) != len(embeddings) or len(ids) != len(texts):
            print("[QdrantVectorStore] ERROR: ids, embeddings, and texts must have the same length")
            return False

        if len(ids) == 0:
            print("[QdrantVectorStore] WARNING: No documents to upsert")
            return True  # Nothing to do, but not an error

        # Check embedding dimensions
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self._vector_dimension:
                print(f"[QdrantVectorStore] ERROR: Embedding {i} has dimension {len(embedding)}, "
                      f"expected {self._vector_dimension}")
                return False

        # =====================================================================
        # STEP 2: Prepare the points for Qdrant
        # =====================================================================
        # Convert our data into Qdrant's PointStruct format.
        # Each point contains an ID, a vector, and a payload (metadata).
        #
        # IMPORTANT: Qdrant requires point IDs to be either:
        # - Unsigned integers (uint64)
        # - UUIDs (as strings in UUID format)
        #
        # It does NOT accept arbitrary strings like "doc_001".
        # So we generate a UUID for the point ID and store the original
        # document ID inside the payload for later retrieval.

        points = []

        for i, (doc_id, embedding, text) in enumerate(zip(ids, embeddings, texts)):
            # Build the payload (metadata stored with the vector)
            # We always include the text, plus any additional metadata
            payload = {
                "text": text,  # Store the original text for retrieval
                "doc_id": doc_id,  # Store the original document ID in payload
            }

            # Add any extra metadata provided by the user
            if metadata and i < len(metadata) and metadata[i]:
                # Merge user-provided metadata into the payload
                payload.update(metadata[i])

            # Generate a UUID for the Qdrant point ID
            # This is required because Qdrant only accepts UUIDs or integers,
            # not arbitrary strings like "doc_001"
            point_uuid = str(uuid.uuid4())

            # Create the PointStruct
            # - id: UUID generated for Qdrant (required format)
            # - vector: The embedding
            # - payload: Text + original doc_id + metadata
            point = PointStruct(
                id=point_uuid,  # Use UUID for Qdrant point ID
                vector=embedding,
                payload=payload
            )
            points.append(point)

        # =====================================================================
        # STEP 3: Upsert the points into Qdrant
        # =====================================================================
        # The upsert operation will:
        # - Insert new points (if ID doesn't exist)
        # - Update existing points (if ID already exists)

        try:
            print(f"[QdrantVectorStore] Upserting {len(points)} points...")

            self._client.upsert(
                collection_name=self._collection_name,
                points=points
            )

            print(f"[QdrantVectorStore] Successfully upserted {len(points)} points!")

            # Print details for learning purposes
            for point in points:
                print(f"  - Doc ID: {point.payload['doc_id']} (Qdrant UUID: {point.id[:8]}...)")
                print(f"    Text preview: {point.payload['text'][:50]}...")
                if 'source' in point.payload:
                    print(f"    Source: {point.payload['source']}")

            return True

        except Exception as e:
            print(f"[QdrantVectorStore] ERROR during upsert: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for vectors similar to the query embedding.

        THIS IS THE HEART OF RAG!
        -------------------------
        When a user asks a question:
        1. The question is converted to an embedding (done by pipeline)
        2. This method finds documents with similar embeddings
        3. Those documents become context for answer generation

        HOW SIMILARITY SEARCH WORKS IN QDRANT:
        --------------------------------------
        1. Qdrant receives the query embedding
        2. It computes cosine similarity with ALL stored embeddings
        3. It ranks results by similarity score
        4. It returns the top K most similar documents

        COSINE SIMILARITY SCORES:
        -------------------------
        - 1.0 = Identical meaning (or exactly the same text)
        - 0.8-0.99 = Very similar (likely relevant)
        - 0.5-0.8 = Somewhat similar (might be relevant)
        - Below 0.5 = Probably not related

        Args:
            query_embedding: The embedding of the user's question.
            top_k: Number of results to return.
            filter_metadata: Optional filters to narrow results.

        Returns:
            List of dictionaries with matching documents:
            [
                {
                    "id": "doc_001",
                    "score": 0.89,
                    "text": "Document content...",
                    "metadata": {"source": "file.pdf", ...}
                },
                ...
            ]

        FILTERING EXAMPLE:
        ------------------
        To only search documents from a specific source:
            results = store.search(
                query_embedding=embedding,
                top_k=5,
                filter_metadata={"source": "company_handbook.pdf"}
            )
        """
        # =====================================================================
        # STEP 1: Validate the query embedding
        # =====================================================================

        if len(query_embedding) != self._vector_dimension:
            print(f"[QdrantVectorStore] ERROR: Query embedding has dimension {len(query_embedding)}, "
                  f"expected {self._vector_dimension}")
            return []

        # =====================================================================
        # STEP 2: Build the filter (if provided)
        # =====================================================================
        # Filters allow narrowing search to specific metadata values.
        # For example: only search documents where source="faq.pdf"

        qdrant_filter = None

        if filter_metadata:
            # Build filter conditions for each metadata field
            conditions = []
            for key, value in filter_metadata.items():
                condition = FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                conditions.append(condition)

            # Combine all conditions with AND logic
            qdrant_filter = Filter(must=conditions)
            print(f"[QdrantVectorStore] Applying filter: {filter_metadata}")

        # =====================================================================
        # STEP 3: Perform the similarity search
        # =====================================================================
        # Qdrant's search method does all the heavy lifting:
        # - Computes similarity scores
        # - Ranks results
        # - Applies filters
        # - Returns top K results

        try:
            print(f"[QdrantVectorStore] Searching for {top_k} similar documents...")

            # Note: Newer versions of qdrant-client use query_points() instead of search()
            # query_points returns a QueryResponse with a .points attribute
            search_response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True  # Include text and metadata in results
            )

            # Extract the points from the response
            search_results = search_response.points

            print(f"[QdrantVectorStore] Found {len(search_results)} results")

        except Exception as e:
            print(f"[QdrantVectorStore] ERROR during search: {e}")
            return []

        # =====================================================================
        # STEP 4: Format results for return
        # =====================================================================
        # Convert Qdrant's result format to our standard format.
        # This keeps our interface consistent regardless of which vector DB we use.

        results = []

        for hit in search_results:
            # Extract text and metadata from payload
            payload = hit.payload or {}
            text = payload.pop("text", "")  # Remove text from payload to avoid duplication
            doc_id = payload.pop("doc_id", str(hit.id))  # Get original doc_id, fallback to UUID

            # Build our standard result format
            # We return the original document ID (e.g., "doc_001"), not the Qdrant UUID
            result = {
                "id": doc_id,  # Return original document ID from payload
                "score": hit.score,  # Similarity score (0 to 1 for cosine)
                "text": text,
                "metadata": payload  # Remaining payload is metadata
            }
            results.append(result)

            # Print for learning purposes
            print(f"\n  [Result] ID: {result['id']}")
            print(f"           Score: {result['score']:.4f}")
            print(f"           Text: {text[:80]}..." if len(text) > 80 else f"           Text: {text}")
            if payload:
                print(f"           Metadata: {payload}")

        return results

    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by their IDs.

        WHEN TO USE:
        ------------
        - Removing outdated documents
        - Cleaning up after re-indexing
        - User data deletion requests

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if deletion was successful.
        """
        if not ids:
            print("[QdrantVectorStore] WARNING: No IDs provided for deletion")
            return True

        try:
            print(f"[QdrantVectorStore] Deleting {len(ids)} points...")

            # Qdrant's delete method accepts a list of point IDs
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=ids
            )

            print(f"[QdrantVectorStore] Successfully deleted {len(ids)} points")
            return True

        except Exception as e:
            print(f"[QdrantVectorStore] ERROR during deletion: {e}")
            return False

    def count(self) -> int:
        """
        Count the total number of vectors in the collection.

        Returns:
            The number of stored vectors.
        """
        try:
            # Get collection info which includes point count
            collection_info = self._client.get_collection(self._collection_name)
            count = collection_info.points_count
            print(f"[QdrantVectorStore] Collection '{self._collection_name}' has {count} vectors")
            return count

        except Exception as e:
            print(f"[QdrantVectorStore] ERROR getting count: {e}")
            return 0

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store.

        Returns:
            Dictionary with collection details.
        """
        try:
            collection_info = self._client.get_collection(self._collection_name)

            return {
                "provider": "qdrant",
                "collection_name": self._collection_name,
                "vector_dimension": self._vector_dimension,
                "total_vectors": collection_info.points_count,
                "status": "connected",
                "distance_metric": "cosine"
            }

        except Exception as e:
            return {
                "provider": "qdrant",
                "collection_name": self._collection_name,
                "vector_dimension": self._vector_dimension,
                "status": f"error: {e}",
                "total_vectors": 0
            }

    def delete_collection(self) -> bool:
        """
        Delete the entire collection from Qdrant.

        WARNING: This permanently deletes ALL data in the collection.
        This action cannot be undone.

        WHEN TO USE:
        -------------
        - Deleting a tenant's entire collection
        - Cleaning up test data
        - User requests data deletion (GDPR, etc.)

        Returns:
            True if deletion was successful (or collection didn't exist).
            False if an error occurred during deletion.

        IDEMPOTENT:
        -----------
        If the collection doesn't exist, returns True (no error).
        """
        try:
            # Check if collection exists
            existing_collections = self._client.get_collections().collections
            existing_names = [col.name for col in existing_collections]

            if self._collection_name not in existing_names:
                print(f"[QdrantVectorStore] Collection '{self._collection_name}' does not exist (idempotent)")
                return True

            # Delete the collection
            print(f"[QdrantVectorStore] Deleting collection '{self._collection_name}'...")
            self._client.delete_collection(collection_name=self._collection_name)
            print(f"[QdrantVectorStore] Collection deleted successfully")

            return True

        except Exception as e:
            print(f"[QdrantVectorStore] ERROR during collection deletion: {e}")
            return False
