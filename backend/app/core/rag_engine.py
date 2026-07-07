import logging
import os
from pathlib import Path

import chromadb
import pandas as pd
from fastapi import HTTPException
from openai import OpenAI

from app.models.schemas import VIQMatch

logger = logging.getLogger(__name__)

# System state, populated by initialize_system() during app startup
client = None
chroma_client = None
collection = None


async def initialize_system():
    """Initialize all system components"""
    global client, chroma_client, collection

    try:
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        client = OpenAI(api_key=openai_api_key)

        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="./data/vectordb")

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

    # Try multiple data file paths
    data_paths = [
        Path("./data/SIRE 2.0 VIQ Mapping Context Sheet.csv"),
        Path("./data/training/FINDINGS DATA FOR VIQ ASSISTANT.xlsx"),
        Path("./data/training/AI VIQ REF testing obs12.xlsx"),
        Path("SIRE 2.0 VIQ Mapping Context Sheet.csv")
    ]

    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            break

    if not data_path:
        logger.error("No data file found in any expected location")
        return

    # Load data based on file type
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    # Handle different column names
    viq_col = None
    question_col = None
    vessel_col = None

    for col in df.columns:
        if 'VIQ' in col.upper() and ('NO' in col.upper() or 'NUM' in col.upper()):
            viq_col = col
        elif 'QUESTION' in col.upper() or 'VIQ' in col.upper():
            question_col = col
        elif 'VESSEL' in col.upper() or 'TYPE' in col.upper():
            vessel_col = col

    if not viq_col or not question_col:
        logger.error(f"Required columns not found in {data_path}")
        return

    df = df.dropna(subset=[viq_col, question_col])
    df = df[df['VIQ No.'] != 'VIQ No.']

    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        viq_no = str(row[viq_col]).strip() if pd.notna(row[viq_col]) else ''
        question = str(row[question_col]).strip() if pd.notna(row[question_col]) else ''
        vessel_types = str(row[vessel_col]).strip() if vessel_col and pd.notna(row[vessel_col]) else "All"

        if not viq_no or not question or viq_no == 'nan' or question == 'nan':
            continue

        doc_text = f"VIQ {viq_no}: {question}"

        context_parts = []
        for col in ['INDUSTRY GUIDANCE', 'INSPECTION GUIDANCE', 'EXPECTED EVIDENCE']:
            if col in df.columns and pd.notna(row.get(col)) and str(row.get(col)).strip() != 'nan':
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

    # Generate embeddings using OpenAI
    embeddings = []
    for i, doc in enumerate(documents):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            embeddings.append(response.data[0].embedding)

            # Log progress every 100 documents
            if (i + 1) % 100 == 0:
                logger.info(f"Generated embeddings for {i + 1}/{len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to generate embedding for document {i}: {str(e)}")
            # Skip this document and continue
            continue

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    logger.info(f"Successfully loaded {len(documents)} VIQ questions")


def perform_search(query: str, vessel_type: str, top_k: int, confidence_threshold: float):
    """Search VIQ questions based on an audit finding. Returns a list of VIQMatch."""
    if not collection:
        raise HTTPException(status_code=500, detail="System not initialized")

    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    # Generate query embedding using OpenAI
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    # Prepare vessel type filter
    where_clause = None
    if vessel_type and vessel_type.lower() != "all":
        vt = str(vessel_type).strip()[:50]
        if vt.replace(' ', '').replace('-', '').replace('_', '').isalnum():
            # Use simple equality check instead of $contains
            where_clause = {"vessel_types": vt}

    # First, try to find in training data (findings)
    matches = []
    try:
        training_collection = chroma_client.get_collection("viq_training")
        training_results = training_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3
        )

        if training_results['documents'] and training_results['documents'][0]:
            seen_viq_numbers = set()

            for i, metadata in enumerate(training_results['metadatas'][0]):
                distance = training_results['distances'][0][i]
                similarity_score = 1 - distance
                viq_number = metadata.get('viq_number', 'Unknown')

                if viq_number in seen_viq_numbers or similarity_score < 0.6:
                    continue

                seen_viq_numbers.add(viq_number)

                viq_results = collection.get(
                    where={"viq_number": viq_number},
                    include=['metadatas']
                )

                if viq_results['metadatas']:
                    viq_metadata = viq_results['metadatas'][0]
                    viq_vessel_types = viq_metadata.get('vessel_types', 'All')

                    if vessel_type and vessel_type.lower() != "all":
                        if (vessel_type not in viq_vessel_types and
                                "All" not in viq_vessel_types):
                            continue

                    match = VIQMatch(
                        viq_number=viq_number,
                        question=viq_metadata.get('question', 'Unknown'),
                        vessel_types=viq_vessel_types,
                        similarity_score=round(similarity_score, 4),
                        context=f"Training match: {metadata.get('finding', '')[:100]}..."
                    )
                    matches.append(match)

                    if len(matches) >= top_k:
                        break
    except:
        pass

    # Fallback to normal VIQ question search if no training matches
    if not matches:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Get more results for filtering
                where=where_clause
            )
        except Exception as filter_error:
            logger.warning(f"ChromaDB filtering failed, using manual filtering: {str(filter_error)}")
            # Fallback: Query without filtering and filter manually
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 3
            )

        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                similarity_score = 1 - distance

                # Manual vessel type filtering if ChromaDB filtering failed
                if vessel_type and vessel_type.lower() != "all":
                    doc_vessel_types = metadata.get('vessel_types', 'All')
                    if (vessel_type not in doc_vessel_types and
                            "All" not in doc_vessel_types):
                        continue

                if similarity_score >= confidence_threshold:
                    match = VIQMatch(
                        viq_number=metadata.get('viq_number', 'Unknown'),
                        question=metadata.get('question', 'Unknown'),
                        vessel_types=metadata.get('vessel_types', 'All'),
                        similarity_score=round(similarity_score, 4),
                        context=metadata.get('context', '')
                    )
                    matches.append(match)

                    if len(matches) >= top_k:
                        break

    return matches
