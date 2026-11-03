"""Vercel entrypoint.

Vercel's Python runtime documents a top-level `module:app` entrypoint for a
FastAPI service, so the platform can find the ASGI application without knowing
about the `server` package. Everything else lives in server/app.py; the local and
Docker run commands (`uvicorn server.app:app`) are unchanged.
"""
from server.app import app  # noqa: F401  (ASGI application discovered by Vercel Services)
