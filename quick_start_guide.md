# Curriculum Search Tool - Quick Start Guide

## Getting Started on Raspberry Pi 5

### 1. Installation (5 minutes)
```bash
# Download and run setup script
curl -sSL https://raw.githubusercontent.com/your-repo/curriculum-search/main/setup.sh | bash

# Or manually clone and setup
git clone https://github.com/your-repo/curriculum-search.git
cd curriculum-search
chmod +x setup.sh
./setup.sh
```

### 2. Basic Usage Examples

#### Command Line Queries
```bash
# Basic search
curriculum_search "slowly changing dimensions"

# Detailed results
curriculum_search "customer address changes" --detailed

# Filter by level
curriculum_search "data visualisation" --level 4

# Interactive mode (recommended for teaching)
curriculum_search --interactive
```

#### Example Outputs
```
🎯 Confidence: 0.92

📍 Location: Level 5, Data Engineering
   Module 3: Data Warehousing, Day 2, Morning Session
   
📚 Topic: Slowly Changing Dimensions
   Learning Outcome: Implement SCD Types 1, 2, and 3
```

### 3. During Teaching Sessions

#### Quick Reference Workflow
1. Learner asks: *"When do we cover slowly changing dimensions?"*
2. You type: `curriculum_search "slowly changing dimensions"`
3. Get instant answer: *"Level 5, Module 3, Day 2, Morning Session"*
4. Optional: Use `--detailed` for full context

#### Interactive Mode for Complex Sessions
```bash
curriculum_search --interactive
```
Then type queries as learners ask questions:
- `"customer data changes"`
- `"Power BI formatting"`  
- `"data quality validation"`

### 4. Updating Your Curriculum Data

#### Option A: Edit CSV Directly
```bash
cd ~/curriculum-search
vi data/curriculum.csv  # Your preferred editor
curriculum_search --rebuild-embeddings
```

#### Option B: Use Excel (Recommended)
1. Edit `data/curriculum.xlsx` with your full curriculum
2. Run update script:
```bash
cd ~/curriculum-search
./scripts/update_data.sh
```

### 5. Sample Queries That Should Work

These queries should find relevant content with the sample data:

**Level 5 Data Engineering:**
- "slowly changing dimensions"
- "customer address updates" 
- "SCD Type 2"
- "data warehouse performance"
- "ETL error handling"
- "data lake vs warehouse"

**Level 4 Data Analyst:**  
- "Power BI reports"
- "chart selection"
- "colour theory"
- "data visualisation principles"

**Cross-Module:**
- "data quality"
- "prerequisites for module 3"
- "morning sessions only"

### 6. Troubleshooting

#### Common Issues:
- **"No relevant content found"**: Try broader keywords or check spelling
- **Slow first run**: Model downloads happen once, then cached
- **Low confidence scores**: Add more keywords to your curriculum data

#### Performance Tips:
- Keep terminal open during teaching for faster queries
- Use `--rebuild-embeddings` only after data updates
- Interactive mode is fastest for multiple queries

### 7. Keyboard Shortcuts (Interactive Mode)

- `Ctrl+C`: Exit gracefully
- `quit` or `exit`: Normal exit  
- `--detailed`: Add to any query for full information
- Up arrow: Recall previous queries (standard bash history)

## Teaching Integration Tips

### During Module Planning
```bash
# Check prerequisites for upcoming topics
curriculum_search "prerequisites" --detailed

# Find related topics
curriculum_search "dimensional modelling"
```

### During Live Sessions  
```bash
# Quick lookups while learners work
curriculum_search --interactive

# Then use natural language:
# "when do we cover this?"
# "what are the prerequisites?"
# "where is performance optimisation?"
```

### For Assessment Planning
```bash
# Find all topics in a module
curriculum_search "Module 3" --detailed

# Check learning outcomes
curriculum_search "learning outcome" --detailed
```

## Next Steps

1. **Populate your data**: Add all 7 modules across both levels
2. **Test thoroughly**: Run sample queries for each module  
3. **Integrate workflow**: Use during next teaching session
4. **Iterate**: Add keywords based on actual learner questions
5. **Scale**: Consider web interface for other tutors

The tool is designed to be your teaching companion - quick, accurate, and always available during those "where do we cover X?" moments! 🎓
