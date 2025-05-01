from fastapi import Header, HTTPException
import os

def verify_admin(x_api_key: str = Header(...)):
    admin_key = os.getenv("ADMIN_API_KEY", "admin-secret")
    if x_api_key != admin_key:
        raise HTTPException(status_code=403, detail="Admin privileges required")