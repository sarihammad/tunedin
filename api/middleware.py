from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os

API_KEY_HEADER = "X-API-KEY"
EXPECTED_API_KEY = os.getenv("TUNEDIN_API_KEY", "dev-secret") 

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api") or request.url.path == "/":
            api_key = request.headers.get(API_KEY_HEADER)
            if api_key != EXPECTED_API_KEY:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
        response = await call_next(request)
        return response