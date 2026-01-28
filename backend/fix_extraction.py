#!/usr/bin/env python3
"""
Simple VIQ Question Extractor - Direct extraction from PDF
"""

import PyPDF2
import re
import chromadb
from pathlib import Path
import openai
import asyncio
from app.config import settings

async def extract_and_store_viq():
    """Extract VIQ questions and store in ChromaDB"""
    
    # Initialize ChromaDB
    persist_dir = Path('data/vectordb')
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Reset collection
    try:
        client.delete_collection('viq_questions')
    except:
        pass
    
    collection = client.create_collection('viq_questions')
    
    # Initialize OpenAI
    openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Process PDFs
    pdf_files = [
        'data/documents/SIRE 2.0 Question Library - Part 1 - Chapters 1 to 7 - Version 1.0 (January 2022) 3.pdf',
        'data/documents/SIRE 2.0 Question Library - Part 2 - Chapters 8 to 12 - Version 1.0 (January 2022) 3.pdf'
    ]
    
    all_questions = []
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Simple pattern: VIQ number followed by question until "Short Question Text"
                pattern = r'(\d+\.\d+\.\d+)\.?\s+(.+?)(?=Short Question Text|Vessel Types|\d+\.\d+\.\d+\.|$)'
                matches = re.findall(pattern, text, re.DOTALL)
                
                for viq_num, question_text in matches:
                    # Clean question
                    question = re.sub(r'\s+', ' ', question_text.strip())
                    
                    # Extract until first question mark
                    if '?' in question:
                        question = question.split('?')[0] + '?'
                    
                    # Skip very short questions
                    if len(question) < 20:
                        continue
                    
                    # Determine vessel type from surrounding text
                    vessel_type = 'All'
                    if 'Oil, Chemical, LPG, LNG' in text:
                        vessel_type = 'All'
                    elif 'Oil' in text and 'Chemical' in text:
                        vessel_type = 'All'
                    
                    all_questions.append({
                        'viq_number': viq_num,
                        'question': question,
                        'vessel_type': vessel_type,
                        'source_file': Path(pdf_file).name
                    })
    
    # Remove duplicates
    unique_questions = {}
    for q in all_questions:
        key = q['viq_number']
        if key not in unique_questions or len(q['question']) > len(unique_questions[key]['question']):
            unique_questions[key] = q
    
    final_questions = list(unique_questions.values())
    print(f"Extracted {len(final_questions)} unique questions")
    
    # Store in ChromaDB
    if final_questions:
        # Get embeddings
        texts = [q['question'] for q in final_questions]
        
        # Process in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts
            )
            batch_embeddings = [emb.embedding for emb in response.data]
            all_embeddings.extend(batch_embeddings)
        
        # Prepare data for ChromaDB
        ids = [f"viq_{q['viq_number']}" for q in final_questions]
        documents = [q['question'] for q in final_questions]
        metadatas = [
            {
                'viq_number': q['viq_number'],
                'question': q['question'],
                'vessel_type': q['vessel_type'],
                'source_file': q['source_file'],
                'guidance': '',
                'category': 'General'
            }
            for q in final_questions
        ]
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=all_embeddings
        )
        
        print(f"✅ Stored {len(final_questions)} questions in ChromaDB")
        
        # Test search for VIQ 5.2.1
        for q in final_questions:
            if q['viq_number'] == '5.2.1':
                print(f"\\n✅ Found VIQ 5.2.1:")
                print(f"Question: {q['question']}")
                break
    
    return len(final_questions)

if __name__ == "__main__":
    asyncio.run(extract_and_store_viq())