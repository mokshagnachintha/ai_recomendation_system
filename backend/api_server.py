from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add current directory to sys.path so local imports work on Vercel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import ProductRecommender

app = FastAPI(title="AI Product Recommender")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = ProductRecommender()

class UserRequest(BaseModel):
    preference: str

@app.post("/api/recommend")
async def recommend(request: UserRequest):
    try:
        recommendations = recommender.get_recommendations(request.preference)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

