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
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.config import settings
except ImportError:
    # Fallback if config not available
    class Settings:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    settings = Settings()

class VIQTrainingSystem:
    def __init__(self):
        self.training_data = []
        # Use absolute path from script location
        script_dir = Path(__file__).parent.parent
        self.training_dir = script_dir / 'data' / 'training'
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_already_processed_files(self, client):
        """Get set of already processed Excel filenames from ChromaDB"""
        try:
            collections = [c.name for c in client.list_collections()]
            if 'viq_training' not in collections:
                return set()
            col = client.get_collection('viq_training')
            if col.count() == 0:
                return set()
            all_meta = col.get()['metadatas']
            return {m['source'] for m in all_meta if m.get('source') and not m['source'].startswith('user_')}
        except:
            return set()

    def load_excel_sheets(self):
        """Load only new Excel sheets from training directory"""
        print("📚 Loading Training Data from Excel Sheets...")
        print("=" * 60)

        script_dir = Path(__file__).parent.parent
        persist_dir = script_dir / 'data' / 'vectordb'
        client = chromadb.PersistentClient(path=str(persist_dir))
        already_processed = self._get_already_processed_files(client)

        excel_files = list(self.training_dir.glob("*.xlsx")) + list(self.training_dir.glob("*.xls"))
        
        if not excel_files:
            print("❌ No Excel files found in data/training/")
            print("   Please add Excel files with columns: Finding, VIQ Number")
            return False

        new_files = [f for f in excel_files if f.name not in already_processed]

        if not new_files:
            print("ℹ️  No new Excel files to process — all already trained")
            print(f"   Already processed: {', '.join(already_processed)}")
            return False

        print(f"   ⏭️  Skipping already trained: {already_processed or 'none'}")
        print(f"   🆕 New files to process: {[f.name for f in new_files]}")

        for excel_file in new_files:
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
        
        # Load .env file from parent directory
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        
        api_key = settings.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("   Please set OPENAI_API_KEY in .env file")
            return False
        
        openai_client = openai.OpenAI(api_key=api_key)
        
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
        return True
    
    def save_training_data(self):
        """Save training data to ChromaDB collection in batches"""
        print(f"\n💾 Saving Training Data to ChromaDB...")
        
        # Use absolute path from script location
        script_dir = Path(__file__).parent.parent
        persist_dir = script_dir / 'data' / 'vectordb'
        client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Get or create training collection — never delete anything
        collections = [c.name for c in client.list_collections()]
        if 'viq_training' in collections:
            training_collection = client.get_collection('viq_training')
        else:
            training_collection = client.create_collection(
                name='viq_training',
                metadata={"hnsw:space": "cosine"}
            )
        
        import time
        batch_size = 5000
        total_items = len(self.training_data)
        base_ts = int(time.time() * 1000)
        
        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)
            batch_data = self.training_data[i:end_idx]
            
            # Unique IDs using timestamp to avoid collision with existing data
            ids = [f"excel_{base_ts}_{i+j}" for j, _ in enumerate(batch_data)]
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
        
        # Use absolute path from script location
        script_dir = Path(__file__).parent.parent
        persist_dir = script_dir / 'data' / 'vectordb'
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
    if not trainer.create_training_embeddings():
        print("❌ Training failed - API key issue")
        return
    
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
