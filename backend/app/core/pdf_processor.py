import os
import re
import pandas as pd
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator
import logging
from datetime import datetime
import hashlib
from app.config import settings
from app.models.schemas import VIQDocument

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.processed_file = settings.PDFS_DIR / "processed_viq_questions.csv"
        self.processing_status = {"status": "idle", "progress": 0.0, "message": ""}
    
    async def process_all_documents(self) -> List[VIQDocument]:
        """Process all documents in the data directory"""
        self.processing_status = {"status": "processing", "progress": 0.0, "message": "Starting document processing"}
        
        try:
            all_documents = []
            
            # Process PDF files first
            pdf_docs = await self._process_pdf_files()
            all_documents.extend(pdf_docs)
            self.processing_status["progress"] = 30.0
            self.processing_status["message"] = f"Processed {len(pdf_docs)} questions from PDF files"
            
            # Process text files
            text_docs = await self._process_text_files()
            all_documents.extend(text_docs)
            self.processing_status["progress"] = 60.0
            self.processing_status["message"] = f"Processed {len(text_docs)} questions from text files"
            
            # Process CSV files
            csv_docs = await self._process_csv_files()
            all_documents.extend(csv_docs)
            self.processing_status["progress"] = 80.0
            self.processing_status["message"] = f"Processed {len(csv_docs)} questions from CSV files"
            
            # Remove duplicates and validate
            unique_docs = await self._deduplicate_documents(all_documents)
            
            # Save processed data
            if unique_docs:
                await self._save_processed_data(unique_docs)
            
            self.processing_status = {
                "status": "completed", 
                "progress": 100.0, 
                "message": f"Successfully processed {len(unique_docs)} unique VIQ questions"
            }
            
            logger.info(f"Document processing completed: {len(unique_docs)} documents")
            return unique_docs
            
        except Exception as e:
            self.processing_status = {
                "status": "error", 
                "progress": 0.0, 
                "message": f"Processing failed: {str(e)}"
            }
            logger.error(f"Document processing failed: {str(e)}")
            return []
    
    async def _process_pdf_files(self) -> List[VIQDocument]:
        """Process PDF files containing VIQ questions"""
        documents = []
        
        pdf_files = list(settings.PDFS_DIR.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing PDF file: {pdf_file.name}")
                
                # Extract text from PDF
                text_content = await self._extract_text_from_pdf(pdf_file)
                
                if text_content:
                    # Extract VIQ questions from the text
                    file_docs = await self._extract_viq_questions(text_content, pdf_file.name)
                    documents.extend(file_docs)
                    logger.info(f"Extracted {len(file_docs)} questions from {pdf_file.name}")
                else:
                    logger.warning(f"No text extracted from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_file}: {str(e)}")
        
        return documents
    
    async def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file"""
        try:
            import PyPDF2
            
            text_content = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} of {pdf_path.name}: {str(e)}")
                        continue
            
            logger.info(f"Extracted {len(text_content)} characters from {pdf_path.name}")
            return text_content
            
        except ImportError:
            logger.error("PyPDF2 not available. Install with: pip install PyPDF2")
            return ""
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
            return ""
    
    async def _process_text_files(self) -> List[VIQDocument]:
        """Process text files containing VIQ questions"""
        documents = []
        
        text_files = list(settings.PDFS_DIR.glob("*.txt"))
        logger.info(f"Found {len(text_files)} text files to process")
        
        for txt_file in text_files:
            try:
                logger.info(f"Processing text file: {txt_file.name}")
                
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract VIQ questions using improved regex patterns
                file_docs = await self._extract_viq_questions(content, txt_file.name)
                documents.extend(file_docs)
                
            except Exception as e:
                logger.error(f"Error processing {txt_file}: {str(e)}")
        
        return documents
    
    async def _process_csv_files(self) -> List[VIQDocument]:
        """Process CSV files containing VIQ data"""
        documents = []
        
        csv_files = [f for f in settings.PDFS_DIR.glob("*.csv") 
                    if f.name != "processed_viq_questions.csv"]
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            try:
                logger.info(f"Processing CSV file: {csv_file.name}")
                
                df = pd.read_csv(csv_file)
                file_docs = await self._extract_from_csv(df, csv_file.name)
                documents.extend(file_docs)
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
        
        return documents
    
    async def _extract_viq_questions(self, content: str, source_file: str) -> List[VIQDocument]:
        """Extract VIQ questions from text content using improved patterns"""
        documents = []
        
        try:
            # Multiple regex patterns for SIRE VIQ format
            patterns = [
                # SIRE format: 2.2.1. Question text followed by Short Question Text
                r'(\d+\.\d+\.\d+)\. (.+?)(?=Short Question Text|Vessel Types|\d+\.\d+\.\d+\.|$)',
                # Alternative SIRE format with newlines
                r'(\d+\.\d+\.\d+)\n(.+?)(?=\nShort Question Text|\nVessel Types|\n\d+\.\d+\.\d+|$)',
                # Standard VIQ pattern: 1.1.1 Question text
                r'(\d+\.\d+\.\d+)\s+(.+?)(?=\n\s*\d+\.\d+\.\d+|\n\s*$|$)',
                # Pattern with question numbers in parentheses
                r'\((\d+\.\d+\.\d+)\)\s*(.+?)(?=\n\s*\(\d+\.\d+\.\d+\)|\n\s*$|$)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
                
                for viq_num, question_text in matches:
                    # Clean up the question text
                    question_text = re.sub(r'\s+', ' ', question_text.strip())
                    question_text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', question_text)
                    
                    # Skip very short or invalid questions
                    if len(question_text) < 20 or len(question_text) > settings.MAX_CONTENT_LENGTH:
                        continue
                    
                    # Determine vessel type and extract guidance
                    vessel_type = self._determine_vessel_type(source_file, question_text)
                    guidance = self._extract_guidance(question_text)
                    
                    # Extract chapter and section info
                    chapter, section = self._extract_chapter_section(content, viq_num)
                    
                    doc = VIQDocument(
                        viq_number=viq_num,
                        question=question_text,
                        vessel_type=vessel_type,
                        guidance=guidance,
                        source_file=source_file,
                        chapter=chapter,
                        section=section,
                        metadata={
                            "extraction_method": "regex",
                            "pattern_used": pattern,
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    
                    documents.append(doc)
                
                if documents:  # If we found matches with this pattern, use them
                    break
            
            logger.info(f"Extracted {len(documents)} questions from {source_file}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to extract VIQ questions from {source_file}: {str(e)}")
            return []
    
    async def _extract_from_csv(self, df: pd.DataFrame, source_file: str) -> List[VIQDocument]:
        """Extract VIQ documents from CSV DataFrame"""
        documents = []
        
        try:
            # Map common column names
            column_mapping = {
                'VIQ Number': 'viq_number',
                'VIQ_Number': 'viq_number',
                'VIQ No.': 'viq_number',
                'Question': 'question',
                'VIQ Question': 'question',
                'VIQ QUESTION': 'question',
                'Vessel Type': 'vessel_type',
                'Vessel_Type': 'vessel_type',
                'VESSEL TYPES': 'vessel_type',
                'Guidance': 'guidance',
                'Notes': 'guidance',
                'INDUSTRY GUIDANCE': 'guidance',
                'Chapter': 'chapter',
                'Section': 'section'
            }
            
            # Rename columns
            df_renamed = df.rename(columns=column_mapping)
            
            for _, row in df_renamed.iterrows():
                try:
                    viq_number = str(row.get('viq_number', '')).strip()
                    question = str(row.get('question', '')).strip()
                    
                    # Skip invalid entries
                    if not viq_number or not question or viq_number == 'nan' or question == 'nan':
                        continue
                    
                    if len(question) < 20 or len(question) > settings.MAX_CONTENT_LENGTH:
                        continue
                    
                    vessel_type = str(row.get('vessel_type', 'All')).strip()
                    if vessel_type == 'nan':
                        vessel_type = 'All'
                    
                    guidance = str(row.get('guidance', '')).strip()
                    if guidance == 'nan':
                        guidance = ''
                    
                    chapter = str(row.get('chapter', '')).strip()
                    if chapter == 'nan':
                        chapter = ''
                    
                    section = str(row.get('section', '')).strip()
                    if section == 'nan':
                        section = ''
                    
                    doc = VIQDocument(
                        viq_number=viq_number,
                        question=question,
                        vessel_type=vessel_type,
                        guidance=guidance,
                        source_file=source_file,
                        chapter=chapter,
                        section=section,
                        metadata={
                            "extraction_method": "csv",
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.warning(f"Error processing CSV row: {str(e)}")
                    continue
            
            logger.info(f"Extracted {len(documents)} questions from CSV {source_file}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to extract from CSV {source_file}: {str(e)}")
            return []
    
    def _determine_vessel_type(self, filename: str, content: str) -> str:
        """Determine vessel type from filename or content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Check for specific vessel type indicators
        vessel_indicators = {
            'Oil': ['oil', 'crude', 'petroleum', 'tanker oil', 'oil tanker'],
            'Chemical': ['chemical', 'chem', 'chemical tanker'],
            'LPG': ['lpg', 'liquefied petroleum gas', 'lpg carrier'],
            'LNG': ['lng', 'liquefied natural gas', 'lng carrier']
        }
        
        for vessel_type, indicators in vessel_indicators.items():
            if any(indicator in filename_lower or indicator in content_lower 
                   for indicator in indicators):
                return vessel_type
        
        return 'All'
    
    def _extract_guidance(self, text: str) -> str:
        """Extract guidance or additional context from question text"""
        # Look for guidance patterns
        guidance_patterns = [
            r'Guidance[:\s]+(.+?)(?=\n\n|\n[A-Z]|$)',
            r'Note[:\s]+(.+?)(?=\n\n|\n[A-Z]|$)',
            r'Inspection[:\s]+(.+?)(?=\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in guidance_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ''
    
    def _extract_chapter_section(self, content: str, viq_number: str) -> tuple:
        """Extract chapter and section information"""
        try:
            # Look for chapter/section headers near the VIQ number
            lines = content.split('\n')
            viq_line_idx = -1
            
            for i, line in enumerate(lines):
                if viq_number in line:
                    viq_line_idx = i
                    break
            
            if viq_line_idx >= 0:
                # Look backwards for chapter/section headers
                for i in range(max(0, viq_line_idx - 10), viq_line_idx):
                    line = lines[i].strip()
                    if re.match(r'Chapter\s+\d+', line, re.IGNORECASE):
                        chapter = line
                    elif re.match(r'Section\s+\d+', line, re.IGNORECASE):
                        section = line
                        return chapter if 'chapter' in locals() else '', section
            
            return '', ''
            
        except Exception:
            return '', ''
    
    async def _deduplicate_documents(self, documents: List[VIQDocument]) -> List[VIQDocument]:
        """Remove duplicate documents based on VIQ number and question content"""
        seen = set()
        unique_docs = []
        
        for doc in documents:
            # Create a hash of VIQ number and question for deduplication
            content_hash = hashlib.md5(
                f"{doc.viq_number}_{doc.question}".encode()
            ).hexdigest()
            
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        logger.info(f"Deduplicated {len(documents)} -> {len(unique_docs)} documents")
        return unique_docs
    
    async def _save_processed_data(self, documents: List[VIQDocument]):
        """Save processed documents to CSV"""
        try:
            # Convert to DataFrame
            data = []
            for doc in documents:
                data.append({
                    'viq_number': doc.viq_number,
                    'question': doc.question,
                    'vessel_type': doc.vessel_type,
                    'guidance': doc.guidance or '',
                    'source_file': doc.source_file,
                    'chapter': doc.chapter or '',
                    'section': doc.section or '',
                    'processed_at': datetime.now().isoformat()
                })
            
            df = pd.DataFrame(data)
            
            # Sort by VIQ number
            df = df.sort_values('viq_number')
            
            # Save to CSV
            df.to_csv(self.processed_file, index=False)
            logger.info(f"Saved {len(df)} processed documents to {self.processed_file}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
    
    async def load_processed_documents(self) -> List[VIQDocument]:
        """Load processed documents from CSV"""
        try:
            if not self.processed_file.exists():
                logger.warning("Processed data file not found")
                return []
            
            df = pd.read_csv(self.processed_file)
            documents = []
            
            for _, row in df.iterrows():
                doc = VIQDocument(
                    viq_number=str(row['viq_number']),
                    question=str(row['question']),
                    vessel_type=str(row['vessel_type']),
                    guidance=str(row.get('guidance', '')),
                    source_file=str(row['source_file']),
                    chapter=str(row.get('chapter', '')),
                    section=str(row.get('section', '')),
                    metadata={
                        "loaded_from": "processed_csv",
                        "processed_at": str(row.get('processed_at', ''))
                    }
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} processed documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load processed documents: {str(e)}")
            return []
    
    def get_processing_status(self) -> Dict[str, any]:
        """Get current processing status"""
        return self.processing_status.copy()