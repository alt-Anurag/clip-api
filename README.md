# CLIP-API – Image Recommendation via Supabase + FastAPI + SentenceTransformer

This project is a FastAPI-based backend that uses OpenAI's CLIP model (`clip-ViT-B-32`) to extract image embeddings and retrieve visually similar products stored in a Supabase database. Hosted on Hugging Face Spaces with a minimal Gradio interface to keep the Space alive.

---

## Features

- Accepts an image URL and returns its embedding
- Finds visually similar products using Supabase RPC
- Built-in retry logic for robust database querying
- Minimal Gradio UI to keep the Hugging Face Space running

---

## API Endpoint

### `POST /recommend`

**Request Body:**

```json
{
  "image_url": "https://example.com/image.jpg",
  "match_threshold": 0.65,
  "match_count": 5
}
```

**Response:**

```json
{
  "embedding": [...],
  "products": [...],
  "image_url": "https://example.com/image.jpg"
}
```

---

## Example Usage

```bash
curl -X POST "https://your-space-url.hf.space/recommend" \
-H "Content-Type: application/json" \
-d '{"image_url": "https://example.com/image.jpg"}'
```

---

## Secrets

Make sure to securely set the following secret in your Hugging Face Space:

- `SUPABASE_KEY` – Your Supabase project’s service role key

---

## Tech Stack

- FastAPI
- Supabase (vector RPC)
- SentenceTransformer (`clip-ViT-B-32`)
- PIL, Requests
- Gradio (for uptime)

---

## Author

[Anurag Kumar Jha](https://github.com/alt-Anurag)

---
