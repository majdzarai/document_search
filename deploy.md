Proceed, but apply changes in phases. After each        
  phase, show the diff and wait for approval.
  I'll enter plan mode to explore the codebase and        
  create a detailed implementation plan for your
  multi-tenant MVP deployment.

  ● Entered plan mode
    Claude is now exploring and designing an
    implementation approach.

  ● Let me explore the codebase to understand the
  existing structure and patterns.

  ● 3 Explore agents finished (ctrl+o to expand)
     ├─ Explore API layer structure · 24 tool uses ·      
  57.7k
     tokens
     │  ⎿  Done
     ├─ Explore security and auth files · 31 tool uses ·  

     53.6k tokens
     │  ⎿  Done
     └─ Explore Qdrant and providers · 9 tool uses ·      
  62.2k
     tokens
        ⎿  Done

  ● Now I have a complete picture. Let me create the      
  implementation plan.

  ● Updated plan
    ⎿  Wrote 1 lines to
       ..\..\.claude\plans\tidy-watching-stallman.md      

       # Multi-Tenant RAG API - Implementation Plan       
       ## Overview
       Add simple API key authentication with
       per-client Qdrant collection isolation for
       Railway deployment. Keep it minimal - no RBAC,     

       JWT, rate limiting, or complex features.
       ---
       ## Architecture

  ● I've created a detailed implementation plan. Here's   
  the summary:

    Key Design Decisions:

    1. API Keys in Environment Variable - Format:
  API_KEYS=sk-client1:tenant1,sk-client2:tenant2
      - No database needed
      - Easy to manage in Railway dashboard
    2. Tenant-Scoped Vector Stores - Each tenant gets     
  their own Qdrant collection (rag_tenant_id)
      - Complete data isolation
      - Thread-safe caching
    3. FastAPI Dependency Injection - Clean auth pattern  
  that protects only /query and /ingest endpoints
    4. Backward Compatible - Existing code paths still    
  work if no tenant specified

    Files to modify/create: 10 files total (6
  modifications, 4 new files)

    Authentication flow:
    Request with "Authorization: Bearer sk-xxx"
      → Extract API key
      → Lookup tenant_id from API_KEYS env
      → Use collection "rag_{tenant_id}"
      → Return isolated results

  ● User approved Claude's plan
    ⎿  Plan saved to:
       ~\.claude\plans\tidy-watching-stallman.md · /plan  

        to edit
       Multi-Tenant RAG API - Implementation Plan

       Overview

       Add simple API key authentication with per-client  

        Qdrant collection isolation for Railway
       deployment. Keep it minimal - no RBAC, JWT, rate   

       limiting, or complex features.

       ---
       Architecture

       PHP Client                     RAG API (Railway)   

                     External Services
           │                               │
                          │
           │ Authorization: Bearer sk-abc  │
                          │
           └──────────────────────────────>│
                          │
                                           │ Extract API  

        key               │
                                           │ Look up      

       tenant_id             │
                                           │ Collection:  

        rag_tenant123     │

       │───────────────────────────────>│ Qdrant Cloud    


       │───────────────────────────────>│ OpenRouter      


       │<───────────────────────────────│
           <───────────────────────────────│
                          │
                  JSON Response            │
                          │

       ---
       File Changes Summary






       File: src/security/api_key.py
       Action: Implement
       Description: API key validation & tenant lookup    

       ────────────────────────────────────────
       File: src/api/dependencies.py
       Action: Implement
       Description: FastAPI auth dependency
       ────────────────────────────────────────
       File: src/core/providers.py
       Action: Modify
       Description: Add tenant-scoped vector store cache  

       ────────────────────────────────────────
       File: src/api/routes/query.py
       Action: Modify
       Description: Add auth dependency, pass tenant      

       ────────────────────────────────────────
       File: src/api/routes/ingest.py
       Action: Modify
       Description: Add auth dependency, pass tenant      

       ────────────────────────────────────────
       File: src/rag/pipeline.py
       Action: Modify
       Description: Accept tenant_id parameter
       ────────────────────────────────────────
       File: src/ingestion/service.py
       Action: Modify
       Description: Accept tenant_id parameter
       ────────────────────────────────────────
       File: Dockerfile
       Action: Create
       Description: Railway deployment
       ────────────────────────────────────────
       File: railway.toml
       Action: Create
       Description: Railway configuration
       ────────────────────────────────────────
       File: docs/CLIENT_GUIDE.md
       Action: Create
       Description: PHP integration examples

       ---
       Detailed Changes

       1. src/security/api_key.py - API Key Storage &     

       Validation

       Purpose: Store API keys and map them to tenant     

       IDs

       # Simple in-memory store loaded from environment   

       variable
       # Format:
       API_KEYS="sk-client1:tenant1,sk-client2:tenant2"   


       API_KEYS = {
           "sk-xxx": "tenant_clientname",
           ...
       }

       def validate_api_key(key: str) -> Optional[str]:   

           """Returns tenant_id if valid, None if
       invalid"""

       Why this approach:
       - No database needed
       - Easy to manage via Railway env vars
       - Simple to understand and debug

       ---
       2. src/api/dependencies.py - Auth Dependency       

       Purpose: FastAPI dependency that extracts and      

       validates API key

       async def get_current_tenant(authorization: str =  

        Header(...)) -> str:
           """
           Extract Bearer token from Authorization        
       header
           Validate against API_KEYS
           Return tenant_id or raise 401
           """

       Usage in routes:
       @router.post("/query")
       async def query(request: QueryRequest, tenant_id:  

        str = Depends(get_current_tenant)):
           # tenant_id is now available

       ---
       3. src/core/providers.py - Tenant-Scoped Vector    

       Stores

       Current: Single global _vector_store singleton     


       Change: Add tenant-scoped cache

       _tenant_vector_stores: Dict[str,
       VectorStoreProvider] = {}
       _tenant_lock = threading.Lock()

       def get_vector_store_for_tenant(tenant_id: str)    

       -> VectorStoreProvider:
           """Get or create vector store with collection  

        'rag_{tenant_id}'"""
           if tenant_id not in _tenant_vector_stores:     

               with _tenant_lock:
                   if tenant_id not in
       _tenant_vector_stores:
                       collection_name =
       f"rag_{tenant_id}"
                       _tenant_vector_stores[tenant_id]   

       = create_vector_store_provider(

       collection_name=collection_name
                       )
           return _tenant_vector_stores[tenant_id]        

       Note: Keep existing get_vector_store() for
       backward compatibility

       ---
       4. src/vectorstore/factory.py - Accept Collection  

        Name Parameter

       Current: Reads collection name from config only    


       Change: Accept optional collection_name parameter  


       def create_vector_store_provider(
           provider_name: str = None,
           collection_name: str = None  # NEW: override

       collection name
       ) -> VectorStoreProvider:
           collection = collection_name or
       settings.qdrant_collection_name
           ...

       ---
       5. src/api/routes/query.py - Add Auth & Tenant     


       Changes:
       1. Import auth dependency
       2. Add tenant_id parameter to endpoint
       3. Pass tenant to pipeline

       from src.api.dependencies import
       get_current_tenant

       @router.post("")
       async def query_rag(
           request: QueryRequest,
           tenant_id: str = Depends(get_current_tenant)   

        # NEW
       ) -> QueryResponse:
           pipeline = get_rag_pipeline(tenant_id)  #      

       Pass tenant
           ...

       ---
       6. src/api/routes/ingest.py - Add Auth & Tenant    


       Same pattern as query:
       @router.post("")
       async def ingest_file(
           file: UploadFile,
           tenant_id: str = Depends(get_current_tenant)   
       ) -> IngestResponse:
           service = get_ingestion_service(tenant_id)     

           ...

       ---
       7. src/rag/pipeline.py - Accept Tenant Parameter   


       Changes:
       - __init__ accepts tenant_id
       - Uses get_vector_store_for_tenant(tenant_id)      

       instead of global

       class RAGPipeline:
           def __init__(self, tenant_id: str = None):     

               self.tenant_id = tenant_id
               if tenant_id:
                   self.vector_store =
       get_vector_store_for_tenant(tenant_id)
               else:
                   self.vector_store =
       get_vector_store()  # backward compat

       ---
       8. src/ingestion/service.py - Accept Tenant        
       Parameter

       Same pattern:
       class IngestionService:
           def __init__(self, tenant_id: str = None):     

               if tenant_id:
                   self.vector_store =
       get_vector_store_for_tenant(tenant_id)
               else:
                   self.vector_store =
       get_vector_store()

       ---
       9. Dockerfile - Railway Deployment

       FROM python:3.11-slim

       WORKDIR /app

       COPY requirements.txt .
       RUN pip install --no-cache-dir -r
       requirements.txt

       COPY . .

       EXPOSE 8000

       CMD ["uvicorn", "src.main:app", "--host",
       "0.0.0.0", "--port", "8000"]

       ---
       10. railway.toml - Railway Config

       [build]
       builder = "dockerfile"

       [deploy]
       healthcheckPath = "/health"
       healthcheckTimeout = 100
       restartPolicyType = "on_failure"

       ---
       11. docs/CLIENT_GUIDE.md - PHP Examples

       // Query example
       $ch =
       curl_init('https://your-app.railway.app/query');   

       curl_setopt($ch, CURLOPT_HTTPHEADER, [
           'Content-Type: application/json',
           'Authorization: Bearer sk-your-api-key'        
       ]);
       curl_setopt($ch, CURLOPT_POSTFIELDS,
       json_encode([
           'question' => 'What is RAG?'
       ]));
       $response = curl_exec($ch);

       ---
       Environment Variables for Railway

       # Required
       OPENROUTER_API_KEY=sk-or-v1-xxx

       # Qdrant Cloud
       QDRANT_URL=https://xxx.cloud.qdrant.io
       QDRANT_API_KEY=xxx

       # API Keys (format: key:tenant_id,key:tenant_id)   

       API_KEYS=sk-client1:acme_corp,sk-client2:beta_inc  


       # Optional
       LLM_MODEL=openai/gpt-4o-mini
       EMBEDDING_MODEL=openai/text-embedding-3-small      


       ---
       Endpoints After Implementation






       Endpoint: GET /health
       Auth Required: No
       Description: Health check for Railway
       ────────────────────────────────────────
       Endpoint: POST /query
       Auth Required: Yes (Bearer)
       Description: Query RAG with tenant isolation       
       ────────────────────────────────────────
       Endpoint: POST /api/v1/ingest
       Auth Required: Yes (Bearer)
       Description: Upload documents to tenant
       collection
       ────────────────────────────────────────
       Endpoint: POST /api/v1/ingest/text
       Auth Required: Yes (Bearer)
       Description: Ingest raw text
       ────────────────────────────────────────
       Endpoint: GET /api/v1/ingest/formats
       Auth Required: Yes (Bearer)
       Description: List supported formats
       ---
       Verification Steps

       1. Local Testing:
       # Set env vars
       set API_KEYS=sk-test:test_tenant
       set OPENROUTER_API_KEY=your-key

       # Start server
       uvicorn src.main:app --reload

       # Test without auth (should fail)
       curl -X POST http://localhost:8000/query -H        
       "Content-Type: application/json" -d
       '{"question":"test"}'
       # Expected: 401 Unauthorized

       # Test with auth
       curl -X POST http://localhost:8000/query -H        
       "Content-Type: application/json" -H
       "Authorization: Bearer sk-test" -d
       '{"question":"test"}'
       # Expected: 200 with answer
       2. Verify tenant isolation:
         - Ingest document with tenant A's key
         - Query with tenant B's key - should NOT find    

       document
         - Query with tenant A's key - should find        
       document
       3. Railway deployment:
         - Push to GitHub
         - Connect Railway to repo
         - Set environment variables
         - Deploy and verify health endpoint

       ---
       What This Does NOT Include (As Requested)

       - No Stripe/billing
       - No dashboards
       - No RBAC (all API keys have same permissions)     

       - No rate limiting
       - No JWT/OAuth
       - No user signup
       - No email verification