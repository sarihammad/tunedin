"""
API routes package.
""" 
from fastapi import APIRouter
# from .recommendation import router as recommendation_router

router = APIRouter()
# router.include_router(recommendation_router, prefix="/recommend")