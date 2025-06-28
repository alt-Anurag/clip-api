from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import base64
import requests
from io import BytesIO
import time
import supabase
import os

app = FastAPI()

SUPABASE_URL = "https://jsnbscsxsqrrdgllgttw.supabase.co"
SUPABASE_KEY = SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Set this securely in Render
client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

model = SentenceTransformer("clip-ViT-B-32")


class EmbedRequest(BaseModel):
    image_url: str
    match_threshold: float = 0.65
    match_count: int = 5


@app.post("/recommend")
def recommend(data: EmbedRequest):
    try:
        # --- Image fetch and embedding
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(data.image_url, stream=True, headers=headers, timeout=10)
        if not response.ok:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")

        image = Image.open(BytesIO(response.content)).convert("RGB")
        query_embedding = model.encode(image).tolist()

        # --- Retry RPC logic
        attempt = 0
        max_retries = 3
        while attempt < max_retries:
            try:
                rpc_response = client.rpc("similar_products", {
                    "query_embedding": query_embedding,
                    "match_threshold": data.match_threshold,
                    "match_count": data.match_count
                }).execute()
                if rpc_response.error:
                    raise Exception(rpc_response.error.message)
                break  # success
            except Exception as e:
                attempt += 1
                if attempt == max_retries:
                    raise HTTPException(status_code=500, detail=f"Supabase RPC failed after retries: {e}")
                time.sleep(2)

        return {
            "embedding": query_embedding,
            "products": rpc_response.data,
            "image_url": data.image_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
