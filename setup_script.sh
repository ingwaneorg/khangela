#!/bin/bash

# Curriculum Search Tool - Raspberry Pi Setup Script
# For Ubuntu Server 24.04 on Raspberry Pi 5

set -e  # Exit on any error

echo "🎓 Setting up Curriculum Search Tool on Raspberry Pi 5"
echo "=================================================="

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    echo "❌ This script is designed for Ubuntu Server 24.04"
    exit 1
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and required system packages
echo "🐍 Installing Python and dependencies..."
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    pkg-config \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    gfortran

# Create project directory
PROJECT_DIR="$HOME/curriculum-search"
echo "📁 Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create directory structure
echo "🏗️  Setting up directory structure..."
mkdir -p {data,src,models,scripts,logs}

# Create Python virtual environment
echo "🔧 Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python packages
echo "📚 Installing Python packages..."
pip install \
    pandas==2.0.3 \
    numpy==1.24.3 \
    sentence-transformers==2.2.2 \
    scikit-learn==1.3.0 \
    openpyxl==3.1.2 \
    torch==2.0.1 \
    transformers==4.31.0

# Create requirements.txt for future reference
pip freeze > requirements.txt

# Copy the search engine script
echo "📝 Setting up search engine..."
cat > src/curriculum_search.py << 'EOF'
#!/usr/bin/env python3
"""
Curriculum Search Tool for Apprenticeship Teaching
A semantic search engine to find curriculum content using natural language queries.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class CurriculumSearchEngine:
    """Semantic search engine for curriculum content."""
    
    def __init__(self, data_path: str = "data/curriculum.csv", 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise the search engine.
        
        Args:
            data_path: Path to curriculum CSV file
            model_name: Sentence transformer model to use
        """
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.model = None
        self.df = None
        self.embeddings = None
        self.embeddings_path = Path("models/embeddings.pkl")
        
    def load_data(self) -> None:
        """Load curriculum data from CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} curriculum entries")
        except FileNotFoundError:
            print(f"❌ Error: Could not find {self.data_path}")
            print("Please ensure curriculum.csv exists in the data directory")
            sys.exit(1)
            
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        print(f"🔄 Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Model loaded successfully")
        
    def create_searchable_content(self) -> List[str]:
        """Create searchable text by combining relevant columns."""
        searchable_texts = []
        
        for _, row in self.df.iterrows():
            # Combine multiple fields for richer search context
            content = f"{row['Topic']} {row['Learning_Outcome']} {row['Content_Description']} {row['Keywords']}"
            searchable_texts.append(content.strip())
            
        return searchable_texts
        
    def generate_embeddings(self, force_rebuild: bool = False) -> None:
        """Generate or load embeddings for the curriculum content."""
        
        if self.embeddings_path.exists() and not force_rebuild:
            print("🔄 Loading existing embeddings...")
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            print("✅ Embeddings loaded from cache")
            return
            
        print("🔄 Generating embeddings for curriculum content...")
        searchable_content = self.create_searchable_content()
        
        # Generate embeddings
        self.embeddings = self.model.encode(searchable_content, 
                                          convert_to_tensor=False,
                                          show_progress_bar=True)
        
        # Save embeddings for future use
        self.embeddings_path.parent.mkdir(exist_ok=True)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print("✅ Embeddings generated and cached")
        
    def search(self, query: str, top_k: int = 5, min_confidence: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        Search for curriculum content using semantic similarity.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (result_dict, confidence_score) tuples
        """
        if self.model is None or self.embeddings is None:
            raise ValueError("Model and embeddings must be loaded first")
            
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        
        # Calculate similarities
        similarities = np.dot(query_embedding, self.embeddings.T)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            confidence = similarities[idx]
            if confidence >= min_confidence:
                result_dict = self.df.iloc[idx].to_dict()
                results.append((result_dict, confidence))
                
        return results
        
    def format_result(self, result: Dict, confidence: float, detailed: bool = False) -> str:
        """Format a search result for display."""
        output = []
        
        # Header with confidence
        confidence_emoji = "🎯" if confidence > 0.8 else "📍" if confidence > 0.6 else "🔍"
        output.append(f"{confidence_emoji} Confidence: {confidence:.2f}")
        output.append("")
        
        # Location information
        location = f"Level {result['Level']}, {result['Stream']}"
        module_info = f"{result['Module']}, Day {result['Day']}, {result['Session']} Session"
        output.append(f"📍 Location: {location}")
        output.append(f"   {module_info}")
        output.append("")
        
        # Content information
        output.append(f"📚 Topic: {result['Topic']}")
        output.append(f"   Learning Outcome: {result['Learning_Outcome']}")
        output.append("")
        
        if detailed:
            output.append(f"📝 Content: {result['Content_Description']}")
            output.append("")
            output.append(f"🏷️  Keywords: {result['Keywords']}")
            output.append("")
            
            if pd.notna(result.get('Prerequisites')):
                output.append(f"📋 Prerequisites: {result['Prerequisites']}")
                output.append("")
                
            if pd.notna(result.get('Duration_Hours')):
                output.append(f"⏱️  Duration: {result['Duration_Hours']} hours")
                output.append("")
        
        return "\n".join(output)
        
    def interactive_search(self) -> None:
        """Run interactive search session."""
        print("\n🎓 Curriculum Search Tool")
        print("=" * 50)
        print("Enter your search queries (type 'quit' to exit)")
        print("Add --detailed for more information")
        print("")
        
        while True:
            try:
                query = input("Search> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                    
                if not query:
                    continue
                    
                # Check for detailed flag
                detailed = False
                if '--detailed' in query:
                    detailed = True
                    query = query.replace('--detailed', '').strip()
                    
                # Perform search
                results = self.search(query)
                
                if not results:
                    print(f"❌ No relevant content found for: '{query}'")
                    print("Try different keywords or check spelling")
                    print("")
                    continue
                    
                print(f"\n🔍 Results for: '{query}'")
                print("-" * 40)
                
                for i, (result, confidence) in enumerate(results, 1):
                    print(f"\n[Result {i}]")
                    print(self.format_result(result, confidence, detailed))
                    
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Curriculum Search Tool")
    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed results')
    parser.add_argument('--level', type=int, choices=[4, 5], 
                       help='Filter by apprenticeship level')
    parser.add_argument('--rebuild-embeddings', action='store_true',
                       help='Force rebuild embeddings cache')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive search session')
    
    args = parser.parse_args()
    
    # Initialise search engine
    search_engine = CurriculumSearchEngine()
    
    try:
        # Load data and model
        search_engine.load_data()
        search_engine.load_model()
        search_engine.generate_embeddings(force_rebuild=args.rebuild_embeddings)
        
        # Interactive mode
        if args.interactive or
