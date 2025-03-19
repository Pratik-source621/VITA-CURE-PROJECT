# main.py
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import cohere
from gtts import gTTS
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize services
app = FastAPI()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)

# CORS Setup (for future frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_disease_name(name: str) -> str:
    """Sanitize and validate disease input"""
    clean_name = name.strip().replace("_", " ").lower()
    if not clean_name.replace(" ", "").isalnum():
        raise HTTPException(400, detail="Invalid characters in disease name")
    if len(clean_name) > 50:
        raise HTTPException(400, detail="Disease name too long (max 50 chars)")
    return clean_name

@app.get("/remedies/{disease}")
async def get_remedies(disease: str):
    """
    Get herbal remedies for a disease
    Example: /remedies/Common_Cold
    """
    try:
        clean_disease = validate_disease_name(disease)
        
        # Get disease info
        disease_data = supabase.table("diseases") \
            .select("*") \
            .ilike("name", f"%{clean_disease}%") \
            .execute()
            
        if not disease_data.data:
            raise HTTPException(404, detail=f"Disease '{clean_disease}' not found")
        
        # Get related remedies
        remedies = supabase.table("herbal_remedies") \
            .select("herb_name, preparation, dosage, safety_notes") \
            .eq("disease_id", disease_data.data[0]["id"]) \
            .execute()

        return {
            "disease": clean_disease,
            "description": disease_data.data[0].get("description", ""),
            "remedies": remedies.data
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=f"Database error: {str(e)}")

@app.get("/cohere-summary/{disease}")
async def generate_summary(disease: str):
    """
    Generate AI summary using Cohere
    Example: /cohere-summary/Migraine
    """
    try:
        # Get remedies first
        remedies_data = await get_remedies(disease)
        
        # Build prompt
        prompt = f"""Generate a herbal remedy summary for {disease} using this data:
        
        Description: {remedies_data['description']}
        
        Remedies:
        {[f"- {r['herb_name']}: {r['preparation']} ({r['dosage']})" for r in remedies_data['remedies']]}
        
        Include safety notes. Use simple language and markdown formatting."""
        
        # Get Cohere response
        response = cohere_client.generate(
            model="command",
            prompt=prompt,
            max_tokens=250,
            temperature=0.5
        )
        
        return {
            "summary": response.generations[0].text,
            "original_data": remedies_data
        }

    except cohere.CohereError as ce:
        raise HTTPException(503, detail=f"Cohere API error: {str(ce)}")
    except Exception as e:
        raise HTTPException(500, detail=f"Summary error: {str(e)}")

@app.get("/text-to-speech")
async def text_to_speech(text: str = Query(..., max_length=500)):
    """
    Convert text to audio (MP3)
    Example: /text-to-speech?text=Ginger helps with digestion
    """
    try:
        tts = gTTS(text=text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return Response(
            content=audio_bytes.read(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"}
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Audio error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
