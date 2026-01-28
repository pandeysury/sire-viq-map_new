from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Global variables
client = None
embedding_model = None
chroma_client = None
collection = None

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global client, embedding_model, chroma_client, collection
    
    try:
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found. AI features disabled.")
        else:
            client = OpenAI(api_key=openai_api_key)
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Load VIQ data and create embeddings
        await load_viq_data()
        
        logger.info("System initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

async def load_viq_data():
    """Load VIQ data from CSV and create embeddings"""
    global collection
    
    try:
        collection = chroma_client.get_collection("viq_questions")
        logger.info("Loaded existing VIQ collection")
        return
    except:
        logger.info("Creating new VIQ collection...")
        collection = chroma_client.create_collection(
            name="viq_questions",
            metadata={"hnsw:space": "cosine"}
        )
    
    # Load CSV data
    csv_path = Path("../data/SIRE 2.0 VIQ Mapping Context Sheet.csv")
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['VIQ No.', 'VIQ QUESTION'])
    df = df[df['VIQ No.'] != 'VIQ No.']
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        viq_no = str(row['VIQ No.']).strip()
        question = str(row['VIQ QUESTION']).strip()
        vessel_types = str(row['VESSEL TYPES']).strip() if pd.notna(row['VESSEL TYPES']) else "All"
        
        if not viq_no or not question or viq_no == 'nan' or question == 'nan':
            continue
            
        doc_text = f"VIQ {viq_no}: {question}"
        
        context_parts = []
        for col in ['INDUSTRY GUIDANCE', 'INSPECTION GUIDANCE', 'EXPECTED EVIDENCE']:
            if pd.notna(row.get(col)) and str(row.get(col)).strip() != 'nan':
                context_parts.append(str(row[col]).strip())
        
        context = " ".join(context_parts) if context_parts else ""
        
        documents.append(doc_text)
        metadatas.append({
            "viq_number": viq_no,
            "question": question,
            "vessel_types": vessel_types,
            "context": context
        })
        ids.append(f"viq_{viq_no}_{idx}")
    
    if not documents:
        logger.error("No valid VIQ data found")
        return
    
    logger.info(f"Generating embeddings for {len(documents)} VIQ questions...")
    embeddings = embedding_model.encode(documents).tolist()
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"Successfully loaded {len(documents)} VIQ questions")

@app.get("/")
async def root():
    return {"message": "VIQ AI Matching System is running"}

@app.post("/search", response_model=QueryResponse)
async def search_viq(request: QueryRequest):
    try:
        if not collection:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        query_embedding = embedding_model.encode([request.query]).tolist()[0]
        
        where_clause = None
        if request.vessel_type and request.vessel_type.lower() != "all":
            where_clause = {
                "$or": [
                    {"vessel_types": {"$contains": request.vessel_type}},
                    {"vessel_types": {"$contains": "All"}}
                ]
            }
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k,
            where=where_clause
        )
        
        matches = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                similarity_score = 1 - distance
                
                match = VIQMatch(
                    viq_number=metadata['viq_number'],
                    question=metadata['question'],
                    vessel_types=metadata['vessel_types'],
                    similarity_score=round(similarity_score, 4),
                    context=metadata.get('context', '')
                )
                matches.append(match)
        
        return QueryResponse(
            matches=matches,
            query=request.query,
            vessel_type=request.vessel_type or "All"
        )
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vessel-types")
async def get_vessel_types():
    try:
        if not collection:
            return {"vessel_types": ["All", "Oil", "Chemical", "LPG", "LNG"]}
        
        results = collection.get()
        vessel_types = set(["All"])
        
        for metadata in results['metadatas']:
            types = metadata.get('vessel_types', '')
            if types:
                for vtype in types.split(','):
                    vtype = vtype.strip()
                    if vtype and vtype != 'nan':
                        vessel_types.add(vtype)
        
        return {"vessel_types": sorted(list(vessel_types))}
        
    except Exception as e:
        logger.error(f"Error getting vessel types: {str(e)}")
        return {"vessel_types": ["All", "Oil", "Chemical", "LPG", "LNG"]}

@app.post("/analyze-finding")
async def analyze_finding(request: QueryRequest):
    try:
        search_response = await search_viq(request)
        
        if not client:
            return search_response
        
        prompt = f"""
Analyze this ship audit finding:

Finding: "{request.query}"
Vessel Type: {request.vessel_type}

Based on these VIQ matches, provide a brief analysis:
{json.dumps([match.dict() for match in search_response.matches[:3]], indent=2)}

Provide concise insights about the most relevant VIQ questions.
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
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
    try:
        if not collection:
            return {"total_questions": 0, "status": "not_initialized"}
        
        count = collection.count()
        return {
            "total_questions": count,
            "status": "ready",
            "embedding_model": "all-MiniLM-L6-v2",
            "database": "ChromaDB"
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"total_questions": 0, "status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)