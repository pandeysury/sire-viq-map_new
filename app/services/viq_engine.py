"""
VIQ business logic — extracted from app/main.py verbatim (structural move
only, no behavior change) except for one deliberate safety fix noted below.

Every function here reads/writes shared handles through `app.state`
(imported as a module, never `from app.state import x`) so all VIQ code
paths — whether reached via the API router or called internally — see
the same OpenAI client, Chroma client and collection.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app import state
from app.schemas import QueryRequest, QueryResponse, VIQMatch
from app.database import SearchHistory

logger = logging.getLogger(__name__)


async def load_viq_data():
    """Load VIQ data from CSV/XLSX and create embeddings on first startup.
    If the viq_questions collection already exists, just attach to it."""
    try:
        state.collection = state.viq_chroma_client.get_collection("viq_questions")
        logger.info("Loaded existing VIQ collection")
        return
    except Exception:
        logger.info("Creating new VIQ collection...")
        state.collection = state.viq_chroma_client.create_collection(
            name="viq_questions",
            metadata={"hnsw:space": "cosine"}
        )

    # Try multiple data file paths
    data_paths = [
        Path("./data/SIRE 2.0 VIQ Mapping Context Sheet.csv"),
        Path("./data/viq_training/FINDINGS DATA FOR VIQ ASSISTANT.xlsx"),
        Path("./data/viq_training/AI VIQ REF testing obs12.xlsx"),
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
            response = state.client.embeddings.create(
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

    state.collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    logger.info(f"Successfully loaded {len(documents)} VIQ questions")


async def search_viq(request: QueryRequest, db: Session, skip_logging: bool = False) -> QueryResponse:
    """Search VIQ questions based on an audit finding.

    `skip_logging=True` is used when this is called internally by
    analyze_finding (which does its own single log write after adding the
    AI analysis) — this replaces the original module-level `_skip_logging`
    global flag with an explicit parameter. The global flag was a latent
    race condition: two concurrent /analyze-finding requests could
    interleave and cause one request's search to skip logging when it
    shouldn't have, or vice versa. An explicit parameter has no such
    cross-request interference and is otherwise behaviorally identical."""
    start_time = datetime.now()

    try:
        if not state.collection:
            raise HTTPException(status_code=500, detail="System not initialized")

        if not state.client:
            raise HTTPException(status_code=500, detail="OpenAI client not initialized")

        # Generate query embedding using OpenAI
        response = state.client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        )
        query_embedding = response.data[0].embedding

        # Prepare vessel type filter
        where_clause = None
        if request.vessel_type and request.vessel_type.lower() != "all":
            vessel_type = str(request.vessel_type).strip()[:50]
            if vessel_type.replace(' ', '').replace('-', '').replace('_', '').isalnum():
                # Use simple equality check instead of $contains
                where_clause = {"vessel_types": vessel_type}

        # First, try to find in training data (findings)
        matches = []
        try:
            training_collection = state.viq_chroma_client.get_collection("viq_training")
            training_results = training_collection.query(
                query_embeddings=[query_embedding],
                n_results=request.top_k * 3
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

                    viq_results = state.collection.get(
                        where={"viq_number": viq_number},
                        include=['metadatas']
                    )

                    if viq_results['metadatas']:
                        viq_metadata = viq_results['metadatas'][0]
                        viq_vessel_types = viq_metadata.get('vessel_types', 'All')

                        if request.vessel_type and request.vessel_type.lower() != "all":
                            if (request.vessel_type not in viq_vessel_types and
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

                        if len(matches) >= request.top_k:
                            break
        except Exception:
            pass

        # Fallback to normal VIQ question search if no training matches
        if not matches:
            try:
                results = state.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=request.top_k * 2,  # Get more results for filtering
                    where=where_clause
                )
            except Exception as filter_error:
                logger.warning(f"ChromaDB filtering failed, using manual filtering: {str(filter_error)}")
                # Fallback: Query without filtering and filter manually
                results = state.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=request.top_k * 3
                )

            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    similarity_score = 1 - distance

                    # Manual vessel type filtering if ChromaDB filtering failed
                    if request.vessel_type and request.vessel_type.lower() != "all":
                        doc_vessel_types = metadata.get('vessel_types', 'All')
                        if (request.vessel_type not in doc_vessel_types and
                                "All" not in doc_vessel_types):
                            continue

                    if similarity_score >= request.confidence_threshold:
                        match = VIQMatch(
                            viq_number=metadata.get('viq_number', 'Unknown'),
                            question=metadata.get('question', 'Unknown'),
                            vessel_types=metadata.get('vessel_types', 'All'),
                            similarity_score=round(similarity_score, 4),
                            context=metadata.get('context', '')
                        )
                        matches.append(match)

                        if len(matches) >= request.top_k:
                            break

        # Log search to database
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        if matches and not skip_logging:
            search_log = SearchHistory(
                query=request.query,
                result_viq=matches[0].viq_number,
                confidence=matches[0].similarity_score,
                time_ms=processing_time,
                vessel_type=request.vessel_type or "All",
                search_method=request.search_method or "hybrid"
            )
            db.add(search_log)
            db.commit()

        return QueryResponse(
            matches=matches,
            query=request.query,
            vessel_type=request.vessel_type or "All"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def analyze_finding(request: QueryRequest, db: Session) -> QueryResponse:
    """Analyze finding with AI enhancement — runs search_viq (without
    logging it twice) then asks GPT-4o-mini to explain the best match."""
    try:
        search_response = await search_viq(request, db, skip_logging=True)

        if not state.client or not search_response.matches:
            return search_response

        # Create focused analysis prompt for ONLY the best match
        best_match = search_response.matches[0]

        prompt = f"""
Analyze this ship audit finding and explain why the best matched VIQ question is relevant:

Finding: "{request.query}"
Vessel Type: {request.vessel_type or 'All'}

Best Match:
VIQ {best_match.viq_number}: {best_match.question}
Vessel Types: {best_match.vessel_types}
Confidence Score: {best_match.similarity_score}

Provide a complete analysis that:
1. Explains what this specific finding indicates about the ship's operations
2. Why this specific VIQ question is the most relevant match
3. What compliance or safety concerns this finding highlights

Provide complete analysis without truncation.
"""

        try:
            response = state.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2
            )

            analysis = response.choices[0].message.content

            if search_response.matches:
                search_response.matches[0].context = f"AI Analysis: {analysis}\\n\\n{search_response.matches[0].context}"

        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {str(e)}")

        # NOTE: preserved from the original behavior — analyze_finding calls
        # search_viq with skip_logging=True (was `_skip_logging = True` in
        # the original) and never logs separately itself, so no
        # SearchHistory row is written for /api/v1/analyze-finding
        # requests, exactly as before.
        return search_response

    except Exception as e:
        logger.error(f"Error in analyze_finding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
