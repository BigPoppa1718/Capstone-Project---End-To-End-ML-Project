#!/bin/bash

# Exit instantly if any single step returns a crash/error code
set -e

# Clear terminal screen layout for clean tracking logs
clear

echo "====================================================================="
echo "🚀 INITIALIZING END-TO-END CAPSTONE MLOPS RUNNER INTERFACE"
echo "====================================================================="

# 1. Environment Verification Layer
echo "🤖 Step 1: Checking Python Virtual Environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✔️ Virtual environment activated successfully."
else
    echo "⚠️ Error: Python venv directory not found! Please create it first."
    exit 1
fi

# 2. Data Engineering Execution
echo -e "\n🩺 Step 2: Launching Step 1 Preprocessing Data Generation Pipeline..."
python src/preprocess.py

# 3. Multi-Model Matrix Grid Optimization
echo -e "\n📈 Step 3: Launching Step 2 Multi-Model MLflow Experiment Matrix..."
python src/train.py

# 4. Independent Report Evaluation Exporter
echo -e "\n📊 Step 4: Compiling Holdout Matrix Metrics & JSON Report Serialization..."
python src/evaluate.py

# 5. Programmatic Champion Leaderboard Picker
echo -e "\n🏆 Step 5: Executing Programmatic Leaderboard Comparison Check..."
python compare_experiments.py

# 6. Automated Unit Verification Testing
echo -e "\n🧪 Step 6: Running Automated pytest Validation Suites Matrix..."
python -m pytest tests/ -v

echo "====================================================================="
echo "🎉 PIPELINE COMPLETE: Your end-to-end ecosystem runs flawlessly!"
echo "====================================================================="
