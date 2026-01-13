"""
Vector Store Base Class
========================

WHAT IS A VECTOR DATABASE?
--------------------------
A vector database (also called a "vector store") is a specialized database designed
to store and search through VECTORS (lists of numbers).

Think of it like this:
- A regular database stores text, numbers, dates, etc.
- A vector database stores EMBEDDINGS (lists of hundreds/thousands of numbers)

WHAT PROBLEM DOES IT SOLVE?
---------------------------
Regular databases are great at exact matching:
- "Find all products where name = 'iPhone'"
- "Find all orders where date > 2024-01-01"

But they're NOT good at "similarity" or "meaning-based" search:
- "Find products similar to this description"
- "Find documents that talk about machine learning"

Vector databases solve this by:
1. Storing embeddings (numerical representations of meaning)
2. Using special algorithms to find SIMILAR vectors quickly
3. Returning the most relevant results ranked by similarity

WHAT IS SIMILARITY SEARCH?
--------------------------
Similarity search finds vectors that are "close" to a query vector.

Imagine a 2D graph where each point represents a document:
- Documents about "cats" cluster together in one area
- Documents about "cars" cluster in another area
- A query about "kittens" would be close to the "cats" cluster

In reality, embeddings have 1000+ dimensions (not just 2), but the concept
is the same: similar meanings = close vectors.

HOW IS SIMILARITY MEASURED?
---------------------------
The most common method is "cosine similarity":
- It measures the angle between two vectors
- Score of 1.0 = identical direction (very similar)
- Score of 0.0 = perpendicular (unrelated)
- Score of -1.0 = opposite direction (opposite meaning)

Example:
- "cat" vs "kitten" → similarity ~0.9 (very similar)
- "cat" vs "dog" → similarity ~0.7 (somewhat similar - both pets)
- "cat" vs "quantum physics" → similarity ~0.1 (not related)

WHY DO WE NEED A BASE CLASS (ABSTRACTION)?
------------------------------------------
There are many vector database options:
- Qdrant: Open-source, easy to use, great for learning
- Pinecone: Cloud-native, fully managed
- Weaviate: Feature-rich, supports hybrid search
- pgvector: PostgreSQL extension
- Milvus: Highly scalable

Each has different APIs and code. By creating an ABSTRACT BASE CLASS:
1. We define a STANDARD interface that all providers must follow
2. We can switch between databases WITHOUT changing our RAG code
3. We make testing easier (can create mock implementations)

This is called the "Strategy Pattern" - same interface, different implementations.

WHY DO WE DELAY CHUNKING?
-------------------------
You might notice we don't implement "chunking" (splitting documents into pieces) here.
That's intentional! Here's why:

1. SEPARATION OF CONCERNS: The vector store's job is to STORE and SEARCH, not to
   process documents. Chunking belongs in the ingestion pipeline.

2. FLEXIBILITY: Different use cases need different chunking strategies:
   - Technical docs might chunk by section
   - Chat logs might chunk by message
   - Books might chunk by paragraph

3. STEP-BY-STEP LEARNING: We're building this system incrementally:
   - Step 2: Embeddings (done!)
   - Step 3: Vector Store (this step!)
   - Step 4+: Ingestion pipeline with chunking

For now, we'll use simple, pre-defined documents to demonstrate vector storage.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStoreProvider(ABC):
    """
    Abstract base class defining the interface for vector store providers.

    WHAT IS THIS CLASS?
    -------------------
    This is an abstract class - a "template" or "contract" that defines what
    methods any vector store provider MUST implement.

    You cannot create an instance of this class directly. Instead, you create
    a subclass (like QdrantVectorStore) that implements all the methods.

    WHAT METHODS ARE REQUIRED?
    --------------------------
    1. upsert()     - Insert or update vectors in the database
    2. search()     - Find vectors similar to a query vector
    3. delete()     - Remove vectors from the database
    4. count()      - Count how many vectors are stored
    5. get_info()   - Get information about the vector store

    HOW TO USE (once a provider is implemented):
    --------------------------------------------
        # Create a provider (e.g., Qdrant)
        provider = QdrantVectorStore(collection_name="my_docs")

        # Store some document embeddings
        provider.upsert(
            ids=["doc1", "doc2"],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            texts=["First document", "Second document"],
            metadata=[{"source": "file1.pdf"}, {"source": "file2.pdf"}]
        )

        # Search for similar documents
        results = provider.search(
            query_embedding=[0.1, 0.2, ...],
            top_k=5
        )

        # Results contain the most similar documents
        for result in results:
            print(f"Document: {result['id']}, Score: {result['score']}")

    DESIGN PATTERN:
    ---------------
    This follows the "Strategy Pattern":
    - The interface (VectorStoreProvider) stays the same
    - Different implementations (Qdrant, Pinecone, etc.) have different code
    - The RAG pipeline doesn't care which implementation is used
    """

    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Insert or update vectors in the database.

        WHAT IS UPSERT?
        ---------------
        "Upsert" = Update + Insert
        - If a vector with the given ID doesn't exist → INSERT it
        - If a vector with the given ID already exists → UPDATE it

        This is useful because:
        1. You don't need to check if something exists first
        2. Re-indexing documents is simple (just upsert them again)
        3. It's a common pattern in databases

        WHAT GETS STORED?
        -----------------
        For each document, we store:
        1. id: A unique identifier (like "doc_001" or a UUID)
        2. embedding: The numerical vector (from the embedding model)
        3. text: The original text (so we can return it in search results)
        4. metadata: Extra information (source file, page number, etc.)

        WHY STORE THE TEXT?
        -------------------
        We could look up the text separately, but storing it in the vector DB:
        - Makes retrieval faster (one query gets everything)
        - Keeps data together (no sync issues)
        - Simplifies the architecture

        Args:
            ids: List of unique identifiers for each vector.
                 Example: ["doc_001", "doc_002", "doc_003"]

            embeddings: List of embedding vectors (lists of floats).
                        Each embedding must have the same dimension.
                        Example: [[0.1, 0.2, ...], [0.3, 0.4, ...]]

            texts: List of original text strings.
                   Used for returning readable results.
                   Example: ["First document text", "Second document text"]

            metadata: Optional list of metadata dictionaries.
                      Store any extra info you need (source, date, etc.)
                      Example: [{"source": "file.pdf", "page": 1}, ...]

        Returns:
            True if the upsert was successful, False otherwise.

        EXAMPLE:
        --------
            # Storing 3 documents
            success = provider.upsert(
                ids=["doc1", "doc2", "doc3"],
                embeddings=[
                    [0.1, 0.2, 0.3, ...],  # 1536 numbers for doc1
                    [0.4, 0.5, 0.6, ...],  # 1536 numbers for doc2
                    [0.7, 0.8, 0.9, ...],  # 1536 numbers for doc3
                ],
                texts=[
                    "Document 1 text here",
                    "Document 2 text here",
                    "Document 3 text here",
                ],
                metadata=[
                    {"source": "manual.pdf", "page": 1},
                    {"source": "manual.pdf", "page": 2},
                    {"source": "faq.txt", "page": 1},
                ]
            )

            if success:
                print("Documents stored successfully!")
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find vectors similar to the query vector.

        THIS IS THE CORE OF RAG!
        ------------------------
        This method is where the "magic" happens:
        1. Take the user's question (as an embedding)
        2. Find the most similar document embeddings
        3. Return those documents to use as context

        HOW SIMILARITY SEARCH WORKS:
        ----------------------------
        1. The query embedding is compared to ALL stored embeddings
        2. A similarity score is calculated for each comparison
        3. Results are sorted by score (highest first)
        4. The top K results are returned

        The similarity is usually "cosine similarity":
        - 1.0 = identical vectors
        - 0.0 = completely unrelated
        - Higher = more similar

        FILTERING (OPTIONAL BUT POWERFUL):
        ----------------------------------
        You can filter results based on metadata BEFORE similarity search.
        This is useful for:
        - Multi-tenant systems: "Only search documents for user X"
        - Time filtering: "Only search documents from 2024"
        - Source filtering: "Only search the FAQ documents"

        Args:
            query_embedding: The embedding vector to search for.
                            This is usually the user's question, embedded.
                            Example: [0.1, 0.2, 0.3, ...]

            top_k: How many results to return.
                   Default is 5.
                   More results = more context but slower and more expensive.
                   Example: 3, 5, 10

            filter_metadata: Optional metadata filter.
                            Only return results matching these criteria.
                            Example: {"source": "faq.pdf"}

        Returns:
            A list of dictionaries, each containing:
            - "id": The document's unique identifier
            - "score": The similarity score (higher = more similar)
            - "text": The document's text content
            - "metadata": The document's metadata

            Results are sorted by score, highest first.

            Example return value:
            [
                {
                    "id": "doc_002",
                    "score": 0.92,
                    "text": "RAG combines retrieval with generation...",
                    "metadata": {"source": "rag_guide.pdf", "page": 3}
                },
                {
                    "id": "doc_005",
                    "score": 0.87,
                    "text": "Vector databases enable semantic search...",
                    "metadata": {"source": "vector_db.pdf", "page": 1}
                }
            ]

        EXAMPLE USAGE:
        --------------
            # Simple search (top 5 results)
            results = provider.search(question_embedding)

            # Custom number of results
            results = provider.search(question_embedding, top_k=10)

            # With filtering
            results = provider.search(
                question_embedding,
                top_k=5,
                filter_metadata={"source": "company_handbook.pdf"}
            )

            # Process results
            for result in results:
                print(f"[Score: {result['score']:.2f}] {result['text'][:100]}...")
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors from the database by their IDs.

        WHY DELETE?
        -----------
        You need to delete vectors when:
        - A document is removed from your knowledge base
        - A document is updated (delete old, insert new)
        - Cleaning up test data
        - User requests data deletion (GDPR, etc.)

        Args:
            ids: List of vector IDs to delete.
                 Example: ["doc_001", "doc_002"]

        Returns:
            True if deletion was successful, False otherwise.

        EXAMPLE:
        --------
            # Delete specific documents
            success = provider.delete(["old_doc_1", "old_doc_2"])

            if success:
                print("Documents deleted!")

        NOTE:
        -----
        Deleting a vector does NOT affect other vectors.
        The similarity search will simply no longer find the deleted vectors.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Count the total number of vectors in the store.

        WHY COUNT?
        ----------
        Useful for:
        - Monitoring: "How many documents have we indexed?"
        - Debugging: "Did my upsert work?"
        - Dashboards: Display collection statistics

        Returns:
            The number of vectors stored (integer).
            Example: 1500 (meaning 1500 document chunks are stored)

        EXAMPLE:
        --------
            num_docs = provider.count()
            print(f"We have {num_docs} documents in the knowledge base")
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store configuration.

        WHY GET INFO?
        -------------
        Useful for:
        - Debugging: "What collection am I connected to?"
        - Logging: Record configuration in logs
        - Health checks: Verify the connection is working
        - Admin dashboards: Display system information

        Returns:
            A dictionary with information about the vector store.

            Common fields:
            - "collection_name": Name of the collection
            - "vector_dimension": Size of stored vectors
            - "total_vectors": Number of vectors stored
            - "provider": Name of the provider (e.g., "qdrant")

            Example:
            {
                "provider": "qdrant",
                "collection_name": "rag_documents",
                "vector_dimension": 1536,
                "total_vectors": 2500,
                "status": "connected"
            }

        EXAMPLE:
        --------
            info = provider.get_info()
            print(f"Connected to: {info['collection_name']}")
            print(f"Vectors stored: {info['total_vectors']}")
        """
        pass

    def delete_collection(self) -> bool:
        """
        Delete the entire collection from the vector database.

        WARNING: This permanently deletes ALL data in the collection.
        This action cannot be undone.

        WHY DELETE COLLECTION?
        ----------------------
        Useful for:
        - Multi-tenant cleanup: Removing a tenant's entire collection
        - Test cleanup: Deleting test data after tests
        - User data deletion: GDPR/compliance requests
        - Collection reset: Starting fresh with a new collection

        WHEN TO OVERRIDE:
        -----------------
        Implement this method if your vector store supports collection deletion.
        Default implementation returns False (not supported).

        Returns:
            True if deletion was successful (or collection didn't exist).
            False if deletion is not supported or an error occurred.

        EXAMPLE:
        --------
            # Delete the entire collection
            success = provider.delete_collection()

            if success:
                print("Collection deleted successfully!")
        """
        pass
