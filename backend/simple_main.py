#!/usr/bin/env python3
"""
VIQ AI Matching System - Simplified Backend
Works with system-installed packages
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    logger.error(f"Missing required packages: {e}")
    logger.error("Please install: pip install fastapi uvicorn")
    exit(1)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="VIQ AI Matching System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str
    vessel_type: Optional[str] = "All"
    top_k: Optional[int] = 5

class VIQMatch(BaseModel):
    viq_number: str
    question: str
    vessel_types: str
    similarity_score: float
    context: Optional[str] = None

class QueryResponse(BaseModel):
    matches: List[VIQMatch]
    query: str
    vessel_type: str

# Global data storage
viq_data = []
openai_client = None

def load_openai_client():
    """Initialize OpenAI client if available"""
    global openai_client
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OPENAI_API_KEY not found")
    except ImportError:
        logger.warning("OpenAI package not available")

def load_viq_data():
    """Load VIQ data from processed CSV"""
    global viq_data
    
    # Try processed CSV first
    csv_path = Path("../data/processed_viq_questions.csv")
    if not csv_path.exists():
        logger.error(f"Processed CSV file not found: {csv_path}")
        return
    
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['viq_number', 'question'])
        
        for _, row in df.iterrows():
            viq_no = str(row['viq_number']).strip()
            question = str(row['question']).strip()
            vessel_type = str(row['vessel_type']).strip() if pd.notna(row['vessel_type']) else "All"
            guidance = str(row['guidance']).strip() if pd.notna(row['guidance']) else ""
            
            if viq_no and question and viq_no != 'nan' and question != 'nan':
                viq_data.append({
                    'viq_number': viq_no,
                    'question': question,
                    'vessel_types': vessel_type,
                    'context': guidance
                })
        
        logger.info(f"Loaded {len(viq_data)} VIQ questions")
        
    except Exception as e:
        logger.error(f"Error loading VIQ data: {str(e)}")

def simple_text_similarity(query: str, text: str) -> float:
    """Simple text similarity using word overlap"""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words or not text_words:
        return 0.0
    
    intersection = query_words.intersection(text_words)
    union = query_words.union(text_words)
    
    return len(intersection) / len(union) if union else 0.0

def search_viq_questions(query: str, vessel_type: str = "All", top_k: int = 5) -> List[dict]:
    """Search VIQ questions using simple text matching"""
    if not viq_data:
        return []
    
    results = []
    
    for item in viq_data:
        # Filter by vessel type
        if vessel_type != "All" and vessel_type not in item['vessel_types']:
            continue
        
        # Calculate similarity
        question_sim = simple_text_similarity(query, item['question'])
        context_sim = simple_text_similarity(query, item['context']) * 0.3  # Lower weight for context
        
        total_similarity = question_sim + context_sim
        
        if total_similarity > 0:
            results.append({
                'viq_number': item['viq_number'],
                'question': item['question'],
                'vessel_types': item['vessel_types'],
                'similarity_score': total_similarity,
                'context': item['context']
            })
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_k]

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    logger.info("Starting VIQ AI Matching System...")
    load_openai_client()
    load_viq_data()
    logger.info("System initialized successfully!")

@app.get("/")
async def root():
    return {"message": "VIQ AI Matching System is running", "status": "ready"}

@app.post("/search", response_model=QueryResponse)
async def search_viq(request: QueryRequest):
    """Search for matching VIQ questions"""
    try:
        results = search_viq_questions(request.query, request.vessel_type, request.top_k)
        
        matches = []
        for result in results:
            match = VIQMatch(
                viq_number=result['viq_number'],
                question=result['question'],
                vessel_types=result['vessel_types'],
                similarity_score=round(result['similarity_score'], 4),
                context=result['context']
            )
            matches.append(match)
        
        return QueryResponse(
            matches=matches,
            query=request.query,
            vessel_type=request.vessel_type
        )
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vessel-types")
async def get_vessel_types():
    """Get available vessel types"""
    vessel_types = set(["All"])
    
    for item in viq_data:
        types = item.get('vessel_types', '')
        if types:
            for vtype in types.split(','):
                vtype = vtype.strip()
                if vtype and vtype != 'nan':
                    vessel_types.add(vtype)
    
    return {"vessel_types": sorted(list(vessel_types))}

@app.post("/analyze-finding")
async def analyze_finding(request: QueryRequest):
    """Analyze audit finding using OpenAI if available"""
    try:
        # Get initial search results
        search_response = await search_viq(request)
        
        if not openai_client:
            return search_response
        
        # Enhance with OpenAI analysis
        try:
            prompt = f"""
Analyze this ship audit finding and provide insights:

Finding: "{request.query}"
Vessel Type: {request.vessel_type}

Based on these VIQ matches, provide a brief analysis:
{json.dumps([match.dict() for match in search_response.matches[:3]], indent=2)}

Provide concise insights about the most relevant VIQ questions.
"""
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            analysis = response.choices[0].message.content
            
            if search_response.matches:
                search_response.matches[0].context = f"AI Analysis: {analysis}\n\n{search_response.matches[0].context}"
            
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {str(e)}")
        
        return search_response
        
    except Exception as e:
        logger.error(f"Error in analyze_finding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_questions": len(viq_data),
        "status": "ready" if viq_data else "no_data",
        "openai_available": openai_client is not None,
        "search_method": "text_similarity"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)