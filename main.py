# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import cohere
import os
import re
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

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_disease_name(name: str) -> str:
    """Sanitize and validate disease input"""
    # Replace underscores and hyphens with spaces
    clean_name = re.sub(r'[_-]', ' ', name).strip().lower()
    
    # Allow letters, numbers, and spaces only
    if not re.match(r'^[\w\s-]+$', clean_name):
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
        
        # Get disease info with exact match
        disease_data = supabase.table("diseases") \
            .select("*") \
            .ilike("name", clean_disease) \
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
        
        # Build prompt with proper formatting
        remedy_lines = []
        for r in remedies_data['remedies']:
            line = f"- **{r['herb_name']}**: {r['preparation']} ({r['dosage']})"
            if r['safety_notes']:
                line += f"\n  - Safety Notes: {r['safety_notes']}"
            remedy_lines.append(line)

        prompt = f"""Generate a comprehensive herbal remedy summary for {remedies_data['disease']} using this data:
        
**Description**: {remedies_data['description']}

**Recommended Herbal Remedies**:
{"\n".join(remedy_lines)}

Include key safety considerations and format the response using markdown with clear sections."""

        # Get Cohere response
        response = cohere_client.generate(
            model="command",
            prompt=prompt,
            max_tokens=300,
            temperature=0.5,
            presence_penalty=0.3
        )

        if not response.generations:
            raise HTTPException(500, detail="Empty response from Cohere API")

        return {
            "summary": response.generations[0].text,
            "original_data": remedies_data
        }

    except cohere.CohereError as ce:
        raise HTTPException(503, detail=f"Cohere API error: {str(ce)}")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=f"Summary error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
