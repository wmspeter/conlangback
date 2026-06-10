import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import firebase_admin
from firebase_admin import credentials, db

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

app = FastAPI()

# Enable CORS so your GitHub Pages frontend can talk to this Render backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google GenAI SDK (Requires GEMINI_API_KEY set in Render environment variables)
genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize Firebase Admin SDK
# You will need to generate a Firebase Service Account Key (JSON) and securely 
# load it. On Render, you can store this JSON as a Secret File.
try:
    # Replace 'firebase-adminsdk.json' with the path to your secret file on Render
    cred = credentials.Certificate("firebase-adminsdk.json") 
    
    # Replace with your actual Firebase Realtime Database URL
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://YOUR-PROJECT-ID.firebaseio.com/' 
    })
except Exception as e:
    print(f"Firebase initialization warning: {e}")

# ==========================================
# 2. DATA MODELS & MATH
# ==========================================

class LogographEntry(BaseModel):
    sound: str
    meaning: str
    image: str
    markers: list[str] = [] # Accommodating the grammar markers array

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates the mathematical distance between two concept vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))

# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.post("/add_word")
def add_word(entry: LogographEntry):
    try:
        # Step A: Convert the meaning text into a vector using text-embedding-004
        response = genai_client.models.embed_content(
            model="text-embedding-004",
            contents=entry.meaning
        )
        # Extract the coordinate array
        vector = response.embeddings[0].values
        
        # Step B: Save everything into the Firebase flat ledger
        ref = db.reference("my_personal_lexicon")
        new_word_ref = ref.push()
        new_word_ref.set({
            "sound": entry.sound,
            "meaning": entry.meaning,
            "markers": entry.markers,
            "image": entry.image,
            "vector": vector
        })
        
        return {"status": "success", "id": new_word_ref.key}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search_lexicon(q: str):
    try:
        # Step A: Convert the user's search query into a target vector
        response = genai_client.models.embed_content(
            model="text-embedding-004",
            contents=q
        )
        query_vector = response.embeddings[0].values
        
        # Step B: Retrieve the entire dictionary from Firebase
        ref = db.reference("my_personal_lexicon")
        lexicon = ref.get()
        
        if not lexicon:
            return []
            
        # Step C: Run Cosine Similarity against all entries
        results = []
        for key, data in lexicon.items():
            word_vector = data.get("vector")
            if word_vector:
                similarity_score = cosine_similarity(query_vector, word_vector)
                
                results.append({
                    "id": key,
                    "sound": data.get("sound"),
                    "meaning": data.get("meaning"),
                    "markers": data.get("markers", []),
                    "image": data.get("image"), # Sends the Base64 string back to UI
                    "similarity": similarity_score
                })
                
        # Step D: Sort by highest conceptual match and return top 3
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:3]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
