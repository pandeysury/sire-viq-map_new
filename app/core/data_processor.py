#!/usr/bin/env python3
"""
VIQ Data Processor - Extract and process VIQ questions from PDFs and text files
"""

import os
import csv
import re
import pandas as pd
from pathlib import Path

class VIQDataProcessor:
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.output_file = self.data_dir / "processed_viq_questions.csv"
        
    def process_text_files(self):
        """Process text files containing VIQ questions"""
        questions = []
        
        for txt_file in self.data_dir.glob("*.txt"):
            print(f"Processing: {txt_file.name}")
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract VIQ questions using regex patterns
            # Pattern for VIQ numbers like "1.1.1", "2.3.4", etc.
            viq_pattern = r'(\d+\.\d+\.\d+)\s+(.+?)(?=\d+\.\d+\.\d+|$)'
            matches = re.findall(viq_pattern, content, re.DOTALL)
            
            for viq_num, question_text in matches:
                # Clean up the question text
                question_text = re.sub(r'\s+', ' ', question_text.strip())
                
                # Determine vessel type from filename or content
                vessel_type = self.determine_vessel_type(txt_file.name, question_text)
                
                if len(question_text) > 10:  # Filter out very short matches
                    questions.append({
                        'viq_number': viq_num,
                        'question': question_text,
                        'vessel_type': vessel_type,
                        'guidance': '',
                        'source_file': txt_file.name
                    })
        
        return questions
    
    def process_csv_files(self):
        """Process existing CSV files"""
        questions = []
        
        for csv_file in self.data_dir.glob("*.csv"):
            print(f"Processing CSV: {csv_file.name}")
            
            try:
                df = pd.read_csv(csv_file)
                
                # Map common column names
                column_mapping = {
                    'VIQ Number': 'viq_number',
                    'VIQ_Number': 'viq_number',
                    'Question': 'question',
                    'VIQ Question': 'question',
                    'Vessel Type': 'vessel_type',
                    'Vessel_Type': 'vessel_type',
                    'Guidance': 'guidance',
                    'Notes': 'guidance'
                }
                
                # Rename columns
                df = df.rename(columns=column_mapping)
                
                # Convert to list of dictionaries
                for _, row in df.iterrows():
                    question_data = {
                        'viq_number': str(row.get('viq_number', '')),
                        'question': str(row.get('question', '')),
                        'vessel_type': str(row.get('vessel_type', 'All')),
                        'guidance': str(row.get('guidance', '')),
                        'source_file': csv_file.name
                    }
                    
                    if question_data['question'] and len(question_data['question']) > 10:
                        questions.append(question_data)
                        
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        return questions
    
    def determine_vessel_type(self, filename, content):
        """Determine vessel type from filename or content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        if any(word in filename_lower or word in content_lower for word in ['oil', 'crude', 'petroleum']):
            return 'Oil'
        elif any(word in filename_lower or word in content_lower for word in ['chemical', 'chem']):
            return 'Chemical'
        elif 'lpg' in filename_lower or 'lpg' in content_lower:
            return 'LPG'
        elif 'lng' in filename_lower or 'lng' in content_lower:
            return 'LNG'
        else:
            return 'All'
    
    def save_processed_data(self, questions):
        """Save processed questions to CSV"""
        df = pd.DataFrame(questions)
        
        # Remove duplicates based on VIQ number and question
        df = df.drop_duplicates(subset=['viq_number', 'question'])
        
        # Sort by VIQ number
        df = df.sort_values('viq_number')
        
        # Save to CSV
        df.to_csv(self.output_file, index=False)
        print(f"Saved {len(df)} questions to {self.output_file}")
        
        return df
    
    def process_all(self):
        """Process all data files"""
        print("Starting VIQ data processing...")
        
        all_questions = []
        
        # Process text files
        text_questions = self.process_text_files()
        all_questions.extend(text_questions)
        print(f"Extracted {len(text_questions)} questions from text files")
        
        # Process CSV files
        csv_questions = self.process_csv_files()
        all_questions.extend(csv_questions)
        print(f"Extracted {len(csv_questions)} questions from CSV files")
        
        # Save processed data
        if all_questions:
            df = self.save_processed_data(all_questions)
            
            # Print summary
            print("\n=== Processing Summary ===")
            print(f"Total questions: {len(df)}")
            print(f"Vessel types: {df['vessel_type'].value_counts().to_dict()}")
            print(f"Output file: {self.output_file}")
            
            return df
        else:
            print("No questions found to process!")
            return None

def main():
    processor = VIQDataProcessor()
    processor.process_all()

if __name__ == "__main__":
    main()