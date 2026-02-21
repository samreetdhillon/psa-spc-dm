@echo off
echo ===================================================
echo   PSA-SPC-DM: STARTING RESEARCH PIPELINE
echo ===================================================

echo [1/6] Generating Data and Extracting Features...
python main.py

echo [2/6] Visualizing Feature Separation...
python src/utils/visualize_separation.py

echo [3/6] Training Random Forest Classifier...
python src/models/train.py

echo [4/6] Evaluating Model Robustness (ROC Curve)...
python src/models/evaluate_robustness.py

echo [5/6] Running Blind Discovery Test...
python src/models/blind_test.py

echo [6/6] Executing Final Discovery Run...
python src/final_discovery_run.py

echo ===================================================
echo   PIPELINE COMPLETE - Check results/figures/
echo ===================================================
pause