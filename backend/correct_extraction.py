#!/usr/bin/env python3
"""
Correct VIQ Extraction System
Extracts ONLY valid VIQ questions from SIRE 2.0 PDFs
"""

import PyPDF2
import re
import chromadb
from pathlib import Path
import openai
import sys
sys.path.append('.')
from app.config import settings

def extract_viq_questions():
    """Extract VIQ questions correctly"""
    
    pdf_files = [
        'data/documents/SIRE 2.0 Question Library - Part 1 - Chapters 1 to 7 - Version 1.0 (January 2022) 3.pdf',
        'data/documents/SIRE 2.0 Question Library - Part 2 - Chapters 8 to 12 - Version 1.0 (January 2022) 3.pdf'
    ]
    
    all_questions = {}
    
    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {Path(pdf_file).name}")
        
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Find VIQ pattern: X.Y.Z followed by question
                # Stop at "Short Question Text" or "Vessel Types"
                pattern = r'(\d+\.\d+\.\d+)\.\s+(.+?)(?=\s+Short Question Text|\s+Vessel Types|\s+\d+\.\d+\.\d+\.|$)'
                matches = re.findall(pattern, text, re.DOTALL)
                
                for viq_num, question_text in matches:
                    # Validate VIQ number (chapters 1-12 only)
                    parts = viq_num.split('.')
                    if len(parts) != 3:
                        continue
                    
                    chapter = int(parts[0])
                    if chapter < 1 or chapter > 12:
                        continue
                    
                    # Clean question
                    question = re.sub(r'\s+', ' ', question_text.strip())
                    
                    # Extract until first question mark
                    if '?' in question:
                        question = question.split('?')[0] + '?'
                    
                    # Skip invalid questions
                    if len(question) < 30 or len(question) > 1000:
                        continue
                    
                    # Skip if it's just a header
                    if question.isupper() or question.count('.') > 20:
                        continue
                    
                    # Extract vessel types from surrounding text
                    vessel_type = 'All'
                    vessel_match = re.search(r'Vessel Types\s+([\w\s,]+)', text[text.find(viq_num):text.find(viq_num)+500])
                    if vessel_match:
                        vessel_text = vessel_match.group(1).strip()
                        if 'Oil' in vessel_text and 'Chemical' in vessel_text:
                            vessel_type = 'All'
                        elif 'Oil' in vessel_text:
                            vessel_type = 'Oil'
                        elif 'Chemical' in vessel_text:
                            vessel_type = 'Chemical'
                        elif 'LPG' in vessel_text:
                            vessel_type = 'LPG'
                        elif 'LNG' in vessel_text:
                            vessel_type = 'LNG'
                    
                    # Store only if better than existing
                    if viq_num not in all_questions or len(question) > len(all_questions[viq_num]['question']):
                        all_questions[viq_num] = {
                            'viq_number': viq_num,
                            'question': question,
                            'vessel_type': vessel_type,
                            'source_file': Path(pdf_file).name
                        }
    
    # Convert to list and sort
    questions_list = list(all_questions.values())
    questions_list.sort(key=lambda x: [int(p) for p in x['viq_number'].split('.')])
    
    print(f"\n✅ Extracted {len(questions_list)} valid VIQ questions")
    
    # Show chapter distribution
    chapter_counts = {}
    for q in questions_list:
        chapter = q['viq_number'].split('.')[0]
        chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
    
    print(f"\n📊 Chapter Distribution:")
    for chapter in sorted(chapter_counts.keys(), key=int):
        print(f"   Chapter {chapter}: {chapter_counts[chapter]} questions")
    
    return questions_list

def store_in_chromadb(questions):
    """Store questions in ChromaDB with OpenAI embeddings"""
    
    print(f"\n💾 Storing in ChromaDB...")
    
    # Initialize ChromaDB
    persist_dir = Path('data/vectordb')
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Reset collection
    try:
        client.delete_collection('viq_questions')
        print("   Deleted old collection")
    except:
        pass
    
    collection = client.create_collection(
        name='viq_questions',
        metadata={"hnsw:space": "cosine"}
    )
    print("   Created new collection")
    
    # Initialize OpenAI
    openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Get embeddings in batches
    print("   Generating embeddings...")
    texts = [q['question'] for q in questions]
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
        print(f"   Processed {min(i+batch_size, len(texts))}/{len(texts)}")
    
    # Prepare data
    ids = [f"viq_{q['viq_number']}" for q in questions]
    documents = [q['question'] for q in questions]
    metadatas = [
        {
            'viq_number': q['viq_number'],
            'question': q['question'],
            'vessel_type': q['vessel_type'],
            'source_file': q['source_file'],
            'guidance': '',
            'category': 'General',
            'chapter': q['viq_number'].split('.')[0],
            'section': ''
        }
        for q in questions
    ]
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=all_embeddings
    )
    
    print(f"   ✅ Stored {len(questions)} questions")
    
    return collection

def test_search(collection):
    """Test search functionality"""
    
    print(f"\n🔍 Testing Search...")
    
    # Test 1: Search for VIQ 5.2.1
    results = collection.get(
        where={"viq_number": "5.2.1"},
        include=['metadatas']
    )
    
    if results['metadatas']:
        print(f"\n✅ VIQ 5.2.1:")
        print(f"   {results['metadatas'][0]['question'][:200]}...")
    else:
        print(f"\n❌ VIQ 5.2.1 not found")
    
    # Test 2: Search for emergency fire pump
    openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    query = "emergency fire pump starting procedure"
    
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(f"\n🔍 Search: '{query}'")
    print(f"   Top 3 Results:")
    for i, metadata in enumerate(results['metadatas'][0]):
        print(f"   {i+1}. VIQ {metadata['viq_number']}: {metadata['question'][:100]}...")

if __name__ == "__main__":
    print("🚢 VIQ Extraction System - SIRE 2.0")
    print("=" * 60)
    
    # Extract questions
    questions = extract_viq_questions()
    
    # Store in ChromaDB
    collection = store_in_chromadb(questions)
    
    # Test search
    test_search(collection)
    
    print(f"\n✅ Done! System ready for use.")
