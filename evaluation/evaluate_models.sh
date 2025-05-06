#!/bin/bash
# Script to run model evaluations

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

# Add the project root to Python path
export PYTHONPATH="$PYTHONPATH:$(dirname $SCRIPT_DIR)"

# Check for required packages
python -c "import pandas, matplotlib" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install pandas matplotlib
fi

# Function to display help
show_help() {
    echo "Usage: ./evaluate_models.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  pairwise        Run a pairwise evaluation comparing Gemini and Gemma"
    echo "  pointwise       Run a pointwise evaluation of a single model"
    echo "  summarize       Generate a summary from an existing results file"
    echo "  visualize       Generate charts from an existing pairwise results file"
    echo ""
    echo "Options:"
    echo "  --model MODEL   Specify the model for pointwise evaluation (gemini or gemma)"
    echo "  --limit N       Limit the number of queries to evaluate"
    echo "  --file FILE     Specify an existing results file for summarize/visualize commands"
    echo ""
    echo "Examples:"
    echo "  ./evaluate_models.sh pairwise"
    echo "  ./evaluate_models.sh pairwise --limit 3"
    echo "  ./evaluate_models.sh pointwise --model gemini"
    echo "  ./evaluate_models.sh summarize --file results/pairwise_comparison_20230815_123456.json"
    echo "  ./evaluate_models.sh visualize --file results/pairwise_comparison_20230815_123456.json"
}

# If no arguments, show help
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Main command
COMMAND=$1
shift  # Remove the command from arguments

# Parse options
MODEL=""
LIMIT=""
FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --file)
            FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute the appropriate command
case $COMMAND in
    pairwise)
        # For new evaluations, run without summarize and charts first
        ARGS="--mode pairwise"
        if [ ! -z "$LIMIT" ]; then
            ARGS="$ARGS --limit $LIMIT"
        fi
        echo "Running pairwise evaluation..."
        python run_evaluation.py $ARGS
        
        # Find the most recent results file
        LATEST_FILE=$(ls -t results/pairwise_comparison_*.json 2>/dev/null | head -n1)
        if [ ! -z "$LATEST_FILE" ]; then
            echo "Generating summary for $LATEST_FILE..."
            python run_evaluation.py --mode pairwise --summarize --charts --results-file "$LATEST_FILE"
        else
            echo "No results file found. Cannot generate summary."
        fi
        ;;
        
    pointwise)
        if [ -z "$MODEL" ]; then
            echo "Error: --model is required for pointwise evaluation"
            exit 1
        fi
        
        # For new evaluations, run without summarize first
        ARGS="--mode pointwise --model $MODEL"
        if [ ! -z "$LIMIT" ]; then
            ARGS="$ARGS --limit $LIMIT"
        fi
        echo "Running pointwise evaluation for $MODEL..."
        python run_evaluation.py $ARGS
        
        # Find the most recent results file
        LATEST_FILE=$(ls -t results/pointwise_${MODEL}_*.json 2>/dev/null | head -n1)
        if [ ! -z "$LATEST_FILE" ]; then
            echo "Generating summary for $LATEST_FILE..."
            python run_evaluation.py --mode pointwise --summarize --results-file "$LATEST_FILE"
        else
            echo "No results file found. Cannot generate summary."
        fi
        ;;
        
    summarize)
        if [ -z "$FILE" ]; then
            echo "Error: --file is required for summarize command"
            exit 1
        fi
        echo "Generating summary for $FILE..."
        python run_evaluation.py --mode pointwise --summarize --results-file "$FILE"
        ;;
        
    visualize)
        if [ -z "$FILE" ]; then
            echo "Error: --file is required for visualize command"
            exit 1
        fi
        echo "Generating visualizations for $FILE..."
        python run_evaluation.py --mode pairwise --charts --results-file "$FILE"
        ;;
        
    help)
        show_help
        ;;
        
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo "Done!"