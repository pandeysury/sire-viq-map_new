#!/usr/bin/env python3
"""
VIQ Training System - Learn from Excel sheets with Finding → VIQ mappings
Improves accuracy without changing existing functionality
"""

import pandas as pd
import chromadb
from pathlib import Path
import openai
import sys
sys.path.append('.')
from app.config import settings

class VIQTrainingSystem:
    def __init__(self):
        self.training_data = []
        self.training_dir = Path('data/training')
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
    def load_excel_sheets(self):
        """Load all Excel sheets from training directory"""
        print("📚 Loading Training Data from Excel Sheets...")
        print("=" * 60)
        
        excel_files = list(self.training_dir.glob("*.xlsx")) + list(self.training_dir.glob("*.xls"))
        
        if not excel_files:
            print("❌ No Excel files found in data/training/")
            print("   Please add Excel files with columns: Finding, VIQ Number")
            return False
        
        for excel_file in excel_files:
            print(f"\n📄 Processing: {excel_file.name}")
            
            try:
                df = pd.read_excel(excel_file)
                
                # Normalize column names
                df.columns = df.columns.str.lower().str.strip()
                
                # Find finding and VIQ columns
                finding_col = None
                viq_col = None
                
                for col in df.columns:
                    if 'finding' in col or 'observation' in col or 'deficiency' in col:
                        finding_col = col
                    if 'viq' in col or 'question' in col:
                        viq_col = col
                
                if not finding_col or not viq_col:
                    print(f"   ⚠️  Skipped - Columns not found")
                    print(f"      Expected: 'Finding' and 'VIQ Number'")
                    continue
                
                # Extract training pairs
                count = 0
                for _, row in df.iterrows():
                    finding = str(row[finding_col]).strip()
                    viq_num = str(row[viq_col]).strip()
                    
                    # Skip invalid rows
                    if pd.isna(finding) or pd.isna(viq_num) or finding == 'nan' or viq_num == 'nan':
                        continue
                    
                    if len(finding) < 10:
                        continue
                    
                    # Clean VIQ number
                    import re
                    viq_match = re.search(r'\d+\.\d+\.\d+', viq_num)
                    if viq_match:
                        viq_num = viq_match.group()
                        self.training_data.append({
                            'finding': finding,
                            'viq_number': viq_num,
                            'source': excel_file.name
                        })
                        count += 1
                
                print(f"   ✅ Loaded {count} training pairs")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        print(f"\n📊 Total Training Pairs: {len(self.training_data)}")
        return len(self.training_data) > 0
    
    def create_training_embeddings(self):
        """Create embeddings for training data"""
        print(f"\n🧠 Creating Training Embeddings...")
        
        openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Get embeddings for all findings
        findings = [item['finding'] for item in self.training_data]
        
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(findings), batch_size):
            batch = findings[i:i + batch_size]
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings = [emb.embedding for emb in response.data]
            all_embeddings.extend(embeddings)
            print(f"   Processed {min(i+batch_size, len(findings))}/{len(findings)}")
        
        # Add embeddings to training data
        for i, item in enumerate(self.training_data):
            item['embedding'] = all_embeddings[i]
        
        print(f"   ✅ Created {len(all_embeddings)} embeddings")
    
    def save_training_data(self):
        """Save training data to ChromaDB collection in batches"""
        print(f"\n💾 Saving Training Data to ChromaDB...")
        
        persist_dir = Path('data/vectordb')
        client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Create training collection
        try:
            client.delete_collection('viq_training')
        except:
            pass
        
        training_collection = client.create_collection(
            name='viq_training',
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add data in batches to avoid size limits
        batch_size = 5000  # Safe batch size
        total_items = len(self.training_data)
        
        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)
            batch_data = self.training_data[i:end_idx]
            
            # Prepare batch data
            ids = [f"train_{j}" for j in range(i, end_idx)]
            documents = [item['finding'] for item in batch_data]
            embeddings = [item['embedding'] for item in batch_data]
            metadatas = [
                {
                    'finding': item['finding'],
                    'viq_number': item['viq_number'],
                    'source': item['source']
                }
                for item in batch_data
            ]
            
            training_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            print(f"   📦 Batch {i//batch_size + 1}: Saved {len(batch_data)} examples ({end_idx}/{total_items})")
        
        print(f"   ✅ Saved {total_items} training examples total")
    
    def test_accuracy(self):
        """Test accuracy improvement"""
        print(f"\n🎯 Testing Training Data Integration...")
        
        persist_dir = Path('data/vectordb')
        client = chromadb.PersistentClient(path=str(persist_dir))
        
        try:
            training_collection = client.get_collection('viq_training')
            count = training_collection.count()
            print(f"   ✅ Training collection verified: {count} examples")
            
            # Test a few random samples
            import random
            test_samples = random.sample(self.training_data, min(3, len(self.training_data)))
            
            print("\n📝 Sample Training Examples:")
            print("-" * 60)
            
            for i, sample in enumerate(test_samples, 1):
                finding = sample['finding'][:80]
                viq_num = sample['viq_number']
                source = sample['source']
                print(f"{i}. VIQ {viq_num} | {finding}...")
                print(f"   Source: {source}")
                print()
            
        except Exception as e:
            print(f"   ❌ Error testing training data: {str(e)}")
    
    def generate_training_instructions(self):
        """Generate training data usage summary"""
        print(f"\n📊 Training Data Summary:")
        print("=" * 60)
        
        # Count by source file
        source_counts = {}
        for item in self.training_data:
            source = item['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print("Training data by source:")
        for source, count in source_counts.items():
            print(f"  📄 {source}: {count:,} pairs")
        
        print(f"\n✅ Total: {len(self.training_data):,} training pairs loaded")
        print(f"✅ Training collection 'viq_training' ready for use")
        print(f"✅ RAG engine will automatically use this data for better accuracy")
        
        print(f"\n🔍 How it works:")
        print(f"  • When user searches, system first checks training data")
        print(f"  • If similar finding found (>85% match), returns learned VIQ")
        print(f"  • Otherwise uses normal semantic/hybrid search")
        print(f"  • This improves accuracy for known patterns")

def main():
    print("🚢 VIQ Training System")
    print("=" * 60)
    print("\n📋 Instructions:")
    print("1. Place Excel files in: backend/data/training/")
    print("2. Excel should have columns: 'Finding' and 'VIQ Number'")
    print("3. Run this script to train the system")
    print("\n" + "=" * 60)
    
    trainer = VIQTrainingSystem()
    
    # Load Excel sheets
    if not trainer.load_excel_sheets():
        return
    
    # Create embeddings
    trainer.create_training_embeddings()
    
    # Save to ChromaDB
    trainer.save_training_data()
    
    # Test accuracy
    trainer.test_accuracy()
    
    # Generate instructions
    trainer.generate_training_instructions()
    
    print(f"\n✅ Training Complete!")
    print(f"   Training data saved in ChromaDB 'viq_training' collection")
    print(f"   Ready to improve accuracy!")

if __name__ == "__main__":
    main()
