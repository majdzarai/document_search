"""
Main Entry Point Module
========================

This is the main entry point for the RAG Service API.

WHAT IS AN ENTRY POINT?
-----------------------
An entry point is the file that starts your application.
When you run the application, this is where execution begins.

For FastAPI applications, the entry point typically:
1. Imports the FastAPI app instance
2. Optionally configures runtime settings
3. Exposes the app for the ASGI server (uvicorn)

HOW TO RUN THIS APPLICATION:
----------------------------
There are two ways to run this application:

1. Using uvicorn directly (RECOMMENDED for development):

   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

   Breaking this down:
   - uvicorn: The ASGI server that runs FastAPI apps
   - src.main:app: Module path (src/main.py) and variable name (app)
   - --reload: Auto-restart when code changes (dev only!)
   - --host 0.0.0.0: Listen on all network interfaces
   - --port 8000: Port number to listen on

2. Running this file directly (useful for debugging):

   python -m src.main

   This uses the uvicorn.run() call at the bottom of this file.

WHAT IS UVICORN?
----------------
Uvicorn is an ASGI server implementation.
ASGI (Asynchronous Server Gateway Interface) is the modern Python
standard for async web servers, replacing the older WSGI standard.

Uvicorn is:
- Fast: Built on uvloop and httptools for high performance
- Standard: Implements the ASGI specification
- Production-ready: Used by many production applications

WHY SEPARATE main.py FROM app.py?
---------------------------------
1. Flexibility: app.py can be imported without running the server
2. Testing: Tests can import the app without starting uvicorn
3. Clarity: Keeps server configuration separate from app configuration
4. Deployment: Different deployment methods can use the app differently
"""

# =============================================================================
# IMPORT THE APPLICATION
# =============================================================================
# We import the app instance from the api.app module.
# This app has all routes registered and middleware configured.

from src.api.app import app


# =============================================================================
# EXPOSE THE APP FOR UVICORN
# =============================================================================
# The 'app' variable is what uvicorn looks for when you run:
#   uvicorn src.main:app --reload
#
# By importing it here, we make it available at the module level.
# This is the standard pattern for FastAPI entry points.

# The app is already imported above, so it's available as src.main:app


# =============================================================================
# OPTIONAL: RUN WITH PYTHON DIRECTLY
# =============================================================================
# This block allows running the server with: python -m src.main
# It's useful for:
# - Quick testing
# - Debugging in IDEs
# - Development without typing the full uvicorn command

if __name__ == "__main__":
    # Import uvicorn only when running directly.
    # This avoids importing uvicorn when the module is just imported.
    import uvicorn

    # Print startup message
    print("=" * 60)
    print("RAG Service API - Starting in development mode")
    print("=" * 60)
    print()
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs:  http://localhost:8000/redoc")
    print("Health Check:      http://localhost:8000/health")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    print()

    # Run the uvicorn server
    # These settings are for DEVELOPMENT ONLY
    uvicorn.run(
        # The app location as a string (allows reload to work)
        "src.main:app",

        # Host: 0.0.0.0 means listen on all network interfaces
        # Use "127.0.0.1" to only allow local connections
        host="0.0.0.0",

        # Port: The port number to listen on
        # 8000 is the conventional port for Python web apps in development
        port=8000,

        # Reload: Automatically restart when code changes
        # NEVER use this in production! It's for development only.
        reload=True,

        # Log Level: Controls how much logging output you see
        # Options: critical, error, warning, info, debug, trace
        log_level="info",
    )
