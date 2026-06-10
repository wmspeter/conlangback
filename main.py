import os
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import firebase_admin
from firebase_admin import credentials, db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    cred = credentials.Certificate("firebase-adminsdk.json") 
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://conlang000-default-rtdb.asia-southeast1.firebasedatabase.app/' 
    })
except Exception as e:
    print(f"Firebase initialization warning: {e}")

class LogographEntry(BaseModel):
    sound: str
    meaning: str
    image: str
    markers: list[str] = []

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 0.0 if norm1 == 0 or norm2 == 0 else float(dot_product / (norm1 * norm2))

@app.post("/add_word")
def add_word(entry: LogographEntry):
    try:
        response = genai_client.models.embed_content(
            model="gemini-embedding-2",
            contents=entry.meaning
        )
        vector = response.embeddings[0].values
        
        ref = db.reference("my_personal_lexicon")
        new_word_ref = ref.push()
        new_word_ref.set({
            "sound": entry.sound,
            "meaning": entry.meaning,
            "markers": entry.markers,
            "image": entry.image,
            "vector": vector,
            "added_date": datetime.utcnow().isoformat() # Time tracking added
        })
        return {"status": "success", "id": new_word_ref.key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/edit_word/{word_id}")
def edit_word(word_id: str, entry: LogographEntry):
    try:
        # Re-embed meaning in case it changed
        response = genai_client.models.embed_content(
            model="gemini-embedding-2",
            contents=entry.meaning
        )
        vector = response.embeddings[0].values
        
        ref = db.reference(f"my_personal_lexicon/{word_id}")
        existing_data = ref.get()
        if not existing_data:
            raise HTTPException(status_code=404, detail="Word not found")

        # Keep original date, update the rest
        ref.update({
            "sound": entry.sound,
            "meaning": entry.meaning,
            "markers": entry.markers,
            "image": entry.image,
            "vector": vector
        })
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search_lexicon(q: str, limit: int = 10): # Limit parameter added
    try:
        response = genai_client.models.embed_content(
            model="gemini-embedding-2",
            contents=q
        )
        query_vector = response.embeddings[0].values
        
        ref = db.reference("my_personal_lexicon")
        lexicon = ref.get()
        if not lexicon: return []
            
        results = []
        for key, data in lexicon.items():
            word_vector = data.get("vector")
            if word_vector:
                results.append({
                    "id": key,
                    "sound": data.get("sound"),
                    "meaning": data.get("meaning"),
                    "markers": data.get("markers", []),
                    "image": data.get("image"),
                    "added_date": data.get("added_date", ""),
                    "similarity": cosine_similarity(query_vector, word_vector)
                })
                
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/words")
def get_all_words():
    try:
        ref = db.reference("my_personal_lexicon")
        lexicon = ref.get()
        if not lexicon: return []
        
        results = []
        for key, data in lexicon.items():
            results.append({
                "id": key,
                "sound": data.get("sound"),
                "meaning": data.get("meaning"),
                "markers": data.get("markers", []),
                "image": data.get("image"),
                "added_date": data.get("added_date", "1970-01-01T00:00:00") # Fallback for old words
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
