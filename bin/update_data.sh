#!/bin/bash
# Update curriculum data and rebuild embeddings

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "🔄 Updating curriculum data..."

# If Excel file exists, convert to CSV
if [ -f "data/curriculum.xlsx" ]; then
    echo "📊 Converting Excel to CSV..."
    python3 -c "
import pandas as pd
df = pd.read_excel('data/curriculum.xlsx')
df.to_csv('data/curriculum.csv', index=False)
print('✅ Converted curriculum.xlsx to curriculum.csv')
"
fi

# Rebuild embeddings
echo "🔄 Rebuilding embeddings..."
./curriculum_search --rebuild-embeddings --interactive

echo "✅ Update complete!"
#EOF
