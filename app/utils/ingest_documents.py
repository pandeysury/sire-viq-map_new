#!/usr/bin/env python3
"""
🚢 VIQ Document Ingestion System
Automatically processes PDF, Excel, CSV files and extracts VIQ questions
"""

import os
import sys
import asyncio
import pandas as pd
import PyPDF2
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings
from app.models.schemas import VIQDocument
from app.core.vector_store import VectorStore
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartDocumentIngester:
    def __init__(self):
        self.documents_dir = settings.BASE_DIR / "data" / "documents"
        self.processed_docs = []
        self.stats = {"pdf": 0, "excel": 0, "csv": 0, "total_questions": 0}
        
        # Initialize OpenAI for AI extraction
        self.openai_client = None
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = openai
    
    async def ingest_all_documents(self):
        """Main ingestion function - processes all supported file types"""
        print("🚢 VIQ Document Ingestion Started")
        print(f"📁 Scanning directory: {self.documents_dir}")
        
        if not self.documents_dir.exists():
            print(f"❌ Directory not found: {self.documents_dir}")
            return
        
        # Get all files
        all_files = list(self.documents_dir.glob("*"))
        supported_files = [f for f in all_files if f.suffix.lower() in ['.pdf', '.xlsx', '.xls', '.csv']]
        
        print(f"📋 Found {len(supported_files)} supported files:")
        for file in supported_files:
            print(f"   • {file.name}")
        
        # Process each file type
        for file_path in supported_files:
            try:
                print(f"\n🔄 Processing: {file_path.name}")
                
                if file_path.suffix.lower() == '.pdf':
                    docs = await self.process_pdf(file_path)
                    self.stats["pdf"] += 1
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    docs = await self.process_excel(file_path)
                    self.stats["excel"] += 1
                elif file_path.suffix.lower() == '.csv':
                    docs = await self.process_csv(file_path)
                    self.stats["csv"] += 1
                
                self.processed_docs.extend(docs)
                self.stats["total_questions"] += len(docs)
                print(f"   ✅ Extracted {len(docs)} questions")
                
            except Exception as e:
                print(f"   ❌ Error processing {file_path.name}: {str(e)}")
        
        # Remove duplicates
        unique_docs = self.deduplicate_documents(self.processed_docs)
        print(f"\n🔍 Deduplication: {len(self.processed_docs)} → {len(unique_docs)} unique questions")
        
        # Store in vector database
        if unique_docs:
            await self.store_in_vector_db(unique_docs)
        
        # Print final stats
        self.print_final_stats(unique_docs)
    
    async def process_pdf(self, file_path: Path) -> List[VIQDocument]:
        """Process PDF files with multiple extraction strategies"""
        documents = []
        
        try:
            # Extract text from PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
            
            print(f"   📄 Extracted {len(full_text)} characters from PDF")
            
            # Strategy 1: Pattern Matching
            pattern_docs = self.extract_with_patterns(full_text, file_path.name)
            if pattern_docs:
                print(f"   🎯 Pattern matching found {len(pattern_docs)} questions")
                documents.extend(pattern_docs)
            
            # Strategy 2: AI Extraction (if no patterns found)
            if not pattern_docs and self.openai_client:
                print("   🤖 Using AI extraction...")
                ai_docs = await self.extract_with_ai(full_text, file_path.name)
                documents.extend(ai_docs)
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
        
        return documents
    
    async def process_excel(self, file_path: Path) -> List[VIQDocument]:
        """Process Excel files with smart column detection"""
        documents = []
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                print(f"   📊 Processing sheet: {sheet_name}")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Strategy 1: Structured format detection
                structured_docs = self.extract_structured_excel(df, file_path.name, sheet_name)
                if structured_docs:
                    documents.extend(structured_docs)
                    continue
                
                # Strategy 2: AI extraction from unstructured Excel
                if self.openai_client:
                    text_content = self.excel_to_text(df)
                    ai_docs = await self.extract_with_ai(text_content, f"{file_path.name}:{sheet_name}")
                    documents.extend(ai_docs)
        
        except Exception as e:
            logger.error(f"Excel processing error: {str(e)}")
        
        return documents
    
    async def process_csv(self, file_path: Path) -> List[VIQDocument]:
        """Process CSV files"""
        documents = []
        
        try:
            df = pd.read_csv(file_path)
            
            # Try structured format first
            structured_docs = self.extract_structured_excel(df, file_path.name, "csv")
            if structured_docs:
                documents.extend(structured_docs)
            elif self.openai_client:
                # AI extraction for unstructured CSV
                text_content = self.excel_to_text(df)
                ai_docs = await self.extract_with_ai(text_content, file_path.name)
                documents.extend(ai_docs)
        
        except Exception as e:
            logger.error(f"CSV processing error: {str(e)}")
        
        return documents
    
    def extract_with_patterns(self, text: str, source_file: str) -> List[VIQDocument]:
        """Extract VIQ questions using regex patterns"""
        documents = []
        
        # Improved VIQ patterns for complete extraction
        patterns = [
            # SIRE format: capture until "Short Question Text" or next VIQ
            r'(\d+\.\d+\.\d+)\.?\s+(.+?)(?=\s*Short Question Text|\s*Vessel Types|\s*\d+\.\d+\.\d+\.|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            
            for viq_num, question_text in matches:
                # Clean question text - remove extra content after question mark
                question_text = re.sub(r'\s+', ' ', question_text.strip())
                question_text = question_text.replace('\n', ' ')
                
                # Extract only the actual question (until first ? or until "Short Question Text")
                if '?' in question_text:
                    question_text = question_text.split('?')[0] + '?'
                elif 'Short Question Text' in question_text:
                    question_text = question_text.split('Short Question Text')[0].strip()
                
                # Skip very short questions
                if len(question_text) < 10:
                    continue
                
                # Determine vessel type and category
                vessel_type = self.determine_vessel_type(source_file, question_text)
                category = self.determine_category(question_text)
                
                doc = VIQDocument(
                    viq_number=viq_num,
                    question=question_text,
                    vessel_type=vessel_type,
                    guidance="",
                    source_file=source_file,
                    chapter="",
                    section="",
                    metadata={
                        "extraction_method": "pattern_matching",
                        "category": category,
                        "processed_at": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            if documents:  # If found with this pattern, stop trying others
                break
        
        return documents
    
    def extract_structured_excel(self, df: pd.DataFrame, source_file: str, sheet_name: str) -> List[VIQDocument]:
        """Extract from structured Excel/CSV with column detection"""
        documents = []
        
        # Column mapping for different naming conventions
        column_mappings = {
            'viq_number': ['viq number', 'viq_number', 'viq no', 'viq', 'question number', 'number', 'id'],
            'question': ['question', 'viq question', 'description', 'text', 'question text', 'details'],
            'vessel_type': ['vessel type', 'vessel_type', 'vessel', 'ship type', 'type'],
            'category': ['category', 'section', 'chapter', 'group', 'area'],
            'guidance': ['guidance', 'notes', 'comments', 'remarks', 'industry guidance']
        }
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Find matching columns
        mapped_columns = {}
        for field, possible_names in column_mappings.items():
            for col in df.columns:
                if any(name in col for name in possible_names):
                    mapped_columns[field] = col
                    break
        
        # Must have at least VIQ number and question
        if 'viq_number' not in mapped_columns or 'question' not in mapped_columns:
            return documents
        
        print(f"   🎯 Found structured format with columns: {list(mapped_columns.keys())}")
        
        # Extract documents
        for _, row in df.iterrows():
            try:
                viq_number = str(row[mapped_columns['viq_number']]).strip()
                question = str(row[mapped_columns['question']]).strip()
                
                # Skip invalid rows
                if pd.isna(viq_number) or pd.isna(question) or viq_number == 'nan' or question == 'nan':
                    continue
                
                if len(question) < 10:
                    continue
                
                # Get other fields
                vessel_type = str(row.get(mapped_columns.get('vessel_type', ''), 'All')).strip()
                if vessel_type == 'nan':
                    vessel_type = 'All'
                
                category = str(row.get(mapped_columns.get('category', ''), '')).strip()
                if category == 'nan':
                    category = ''
                
                guidance = str(row.get(mapped_columns.get('guidance', ''), '')).strip()
                if guidance == 'nan':
                    guidance = ''
                
                doc = VIQDocument(
                    viq_number=viq_number,
                    question=question,
                    vessel_type=vessel_type,
                    guidance=guidance,
                    source_file=f"{source_file}:{sheet_name}",
                    chapter="",
                    section="",
                    metadata={
                        "extraction_method": "structured_excel",
                        "category": category,
                        "processed_at": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error processing row: {str(e)}")
                continue
        
        return documents
    
    async def extract_with_ai(self, text: str, source_file: str) -> List[VIQDocument]:
        """Use AI to extract VIQ questions from unstructured text"""
        documents = []
        
        if not self.openai_client:
            return documents
        
        try:
            # Limit text size for API
            if len(text) > 8000:
                text = text[:8000] + "..."
            
            prompt = f"""
Extract VIQ (Vessel Inspection Questionnaire) questions from the following text.
Return a JSON array with this format:
[
  {{
    "viq_number": "7.1.2",
    "question": "Are emergency escape routes clearly marked?",
    "vessel_type": "All",
    "category": "Safety"
  }}
]

Rules:
1. Assign VIQ numbers in format X.Y.Z if not present
2. Vessel types: Oil, Chemical, LPG, LNG, or All
3. Categories: Safety, Navigation, Cargo, Machinery, Documentation, Environmental
4. Extract only actual inspection questions
5. Minimum 10 characters per question

Text:
{text}
"""
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Parse AI response
            ai_result = response.choices[0].message.content
            
            # Try to extract JSON
            import json
            json_match = re.search(r'\[.*\]', ai_result, re.DOTALL)
            if json_match:
                questions_data = json.loads(json_match.group())
                
                for item in questions_data:
                    doc = VIQDocument(
                        viq_number=item.get('viq_number', '0.0.0'),
                        question=item.get('question', ''),
                        vessel_type=item.get('vessel_type', 'All'),
                        guidance="",
                        source_file=source_file,
                        chapter="",
                        section="",
                        metadata={
                            "extraction_method": "ai_extraction",
                            "category": item.get('category', ''),
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
        
        except Exception as e:
            logger.error(f"AI extraction error: {str(e)}")
        
        return documents
    
    def excel_to_text(self, df: pd.DataFrame) -> str:
        """Convert Excel DataFrame to text for AI processing"""
        text_parts = []
        
        for _, row in df.iterrows():
            row_text = " | ".join([str(val) for val in row.values if pd.notna(val)])
            if len(row_text) > 10:
                text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def determine_vessel_type(self, filename: str, content: str) -> str:
        """Determine vessel type from filename or content"""
        text = (filename + " " + content).lower()
        
        # Check for specific vessel type patterns in SIRE format
        if 'vessel types oil, chemical, lpg, lng' in text:
            return 'All'  # Applicable to multiple types
        elif 'vessel types oil' in text and 'chemical' in text and 'lpg' in text:
            return 'All'
        elif 'oil, chemical, lpg, lng' in text:
            return 'All'
        elif 'vessel types oil' in text:
            return 'Oil'
        elif 'vessel types chemical' in text:
            return 'Chemical'
        elif 'vessel types lpg' in text:
            return 'LPG'
        elif 'vessel types lng' in text:
            return 'LNG'
        
        # Fallback to keyword detection
        vessel_indicators = {
            'Oil': ['oil tanker', 'crude oil', 'petroleum'],
            'Chemical': ['chemical tanker', 'chemical'],
            'LPG': ['lpg carrier', 'liquefied petroleum'],
            'LNG': ['lng carrier', 'liquefied natural']
        }
        
        for vessel_type, indicators in vessel_indicators.items():
            if any(indicator in text for indicator in indicators):
                return vessel_type
        
        return 'All'
    
    def determine_category(self, question_text: str) -> str:
        """Determine question category from content"""
        text = question_text.lower()
        
        categories = {
            'Safety': ['safety', 'emergency', 'fire', 'escape', 'alarm', 'drill'],
            'Navigation': ['navigation', 'bridge', 'radar', 'gps', 'chart'],
            'Cargo': ['cargo', 'pump', 'tank', 'loading', 'discharge'],
            'Machinery': ['engine', 'machinery', 'motor', 'generator', 'boiler'],
            'Environmental': ['pollution', 'waste', 'oil spill', 'environment'],
            'Documentation': ['certificate', 'document', 'record', 'log']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'General'
    
    def deduplicate_documents(self, documents: List[VIQDocument]) -> List[VIQDocument]:
        """Remove duplicate documents"""
        seen = set()
        unique_docs = []
        
        for doc in documents:
            # Create unique key
            key = f"{doc.viq_number}_{doc.question[:50]}"
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        return unique_docs
    
    async def store_in_vector_db(self, documents: List[VIQDocument]):
        """Store documents in vector database"""
        try:
            print(f"\n💾 Storing {len(documents)} documents in vector database...")
            
            # Initialize vector store
            vector_store = VectorStore()
            await vector_store.initialize()
            
            # Reset collection
            await vector_store.reset_collection()
            
            # Add documents
            await vector_store.add_documents(documents)
            
            print("   ✅ Successfully stored in vector database")
            
        except Exception as e:
            print(f"   ❌ Vector storage error: {str(e)}")
    
    def print_final_stats(self, documents: List[VIQDocument]):
        """Print final processing statistics"""
        print(f"\n🎉 Document Ingestion Complete!")
        print(f"📊 Processing Statistics:")
        print(f"   • PDF files processed: {self.stats['pdf']}")
        print(f"   • Excel files processed: {self.stats['excel']}")
        print(f"   • CSV files processed: {self.stats['csv']}")
        print(f"   • Total questions extracted: {len(documents)}")
        
        # Vessel type breakdown
        vessel_counts = {}
        category_counts = {}
        
        for doc in documents:
            vessel_counts[doc.vessel_type] = vessel_counts.get(doc.vessel_type, 0) + 1
            category = doc.metadata.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"\n🚢 Vessel Type Distribution:")
        for vessel, count in vessel_counts.items():
            print(f"   • {vessel}: {count} questions")
        
        print(f"\n📋 Category Distribution:")
        for category, count in category_counts.items():
            print(f"   • {category}: {count} questions")
        
        print(f"\n✅ Ready for search! Use the API to query VIQ questions.")

async def main():
    """Main function"""
    ingester = SmartDocumentIngester()
    await ingester.ingest_all_documents()

if __name__ == "__main__":
    asyncio.run(main())