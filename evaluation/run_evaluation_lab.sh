#!/bin/bash
# Script to automate the full evaluation process for the lab
# This script works with the existing L200-upskilling repository structure

echo "======================================================"
echo "Model Evaluation Lab: Automatic Execution Script"
echo "======================================================"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

# Create results directory if it doesn't exist
mkdir -p results

# Step 1: Check dependencies
echo "Step 1: Checking dependencies..."
python -c "import pandas, matplotlib" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install pandas matplotlib
fi
echo "Dependencies check: Complete"
echo ""

# Step 2: Check for directory structure and model access
echo "Step 2: Checking environment setup..."
if [ -f "../app/backend/models/gemini_chat_refactored.py" ] && [ -f "../app/backend/models/gemma_chat.py" ]; then
    echo "✅ Model files found"
else
    echo "⚠️ Warning: Model files not found in the expected location"
    echo "This may cause the evaluation to fail"
    echo "Check that you're running this script from the L200-upskilling/evaluation directory"
fi

if [ -f "data/test_queries.json" ]; then
    echo "✅ Test queries found"
else
    echo "⚠️ Warning: Test queries file not found"
    echo "Please ensure data/test_queries.json exists"
fi
echo ""

# Step 3: Run pointwise evaluation for Gemini
echo "Step 3: Running pointwise evaluation for Gemini model..."
echo "This will test the Gemini model on all test queries and measure performance."
./evaluate_models.sh pointwise --model gemini
echo ""

# Step 4: Run pointwise evaluation for Gemma
echo "Step 4: Running pointwise evaluation for Gemma model..."
echo "This will test the Gemma model on all test queries and measure performance."
./evaluate_models.sh pointwise --model gemma
echo ""

# Step 5: Run pairwise evaluation to compare both models
echo "Step 5: Running pairwise evaluation to compare Gemini and Gemma..."
echo "This will directly compare the two models on the same test queries."
./evaluate_models.sh pairwise
echo ""

# Step 6: Generate comparative visualization
echo "Step 6: Generating comparative visualization..."
LATEST_PAIRWISE=$(ls -t results/pairwise_comparison_*.json 2>/dev/null | head -n1)
if [ ! -z "$LATEST_PAIRWISE" ]; then
    echo "Creating visualizations for $LATEST_PAIRWISE"
    ./evaluate_models.sh visualize --file "$LATEST_PAIRWISE"
    
    # Open the directory with the charts if on a system with a GUI
    CHARTS_DIR=$(echo $LATEST_PAIRWISE | sed 's/\.json//g' | sed 's/pairwise_comparison_/charts_/g')
    if [ -d "results/$CHARTS_DIR" ]; then
        echo "Charts generated in: results/$CHARTS_DIR"
        # Uncomment to open the directory if on a system with a GUI
        # open "results/$CHARTS_DIR"
    fi
else
    echo "No pairwise results found to visualize."
fi
echo ""

# Step 7: Show summary of results
echo "Step 7: Summary of evaluation results..."
LATEST_GEMINI=$(ls -t results/pointwise_gemini_*.json 2>/dev/null | head -n1)
LATEST_GEMMA=$(ls -t results/pointwise_gemma_*.json 2>/dev/null | head -n1)
LATEST_PAIRWISE_SUMMARY=$(ls -t results/summary_*.csv 2>/dev/null | head -n1)

if [ ! -z "$LATEST_GEMINI" ]; then
    echo "Gemini Results: $LATEST_GEMINI"
fi
if [ ! -z "$LATEST_GEMMA" ]; then
    echo "Gemma Results: $LATEST_GEMMA"
fi
if [ ! -z "$LATEST_PAIRWISE_SUMMARY" ]; then
    echo "Pairwise Comparison Summary: $LATEST_PAIRWISE_SUMMARY"
    echo ""
    echo "Displaying summary table:"
    echo "-------------------------"
    head -n 10 "$LATEST_PAIRWISE_SUMMARY"
fi
echo ""

echo "======================================================"
echo "Evaluation process complete!"
echo "======================================================"
echo ""
echo "Next steps:"
echo "1. Review the summary tables in the results/ directory"
echo "2. Examine the generated charts for visual comparisons"
echo "3. Analyze the JSON files for detailed metrics"
echo "4. Explore creating your own test queries (see QUERY_CREATION_GUIDELINES.md)"
echo ""
echo "For more information, see the MODEL_EVALUATION_LAB.md guide" 