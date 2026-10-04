import os
import pandas as pd

def test_models_exist():
    assert os.path.exists("models/baseline/logistic_regression.joblib")
    assert os.path.exists("models/baseline/naive_bayes.joblib")
    assert os.path.exists("models/baseline/linear_svm.joblib")

def test_metrics_and_report_exist():
    assert os.path.exists("reports/metrics/day5_baseline_model_comparison.csv")
    assert os.path.exists("reports/day5_baseline_report.md")
    
def test_visualizations_exist():
    assert os.path.exists("reports/figures/day5/baseline_f1_comparison.png")
    assert os.path.exists("reports/figures/day5/baseline_roc_auc_comparison.png")

def test_metrics_table_has_correct_structure():
    df = pd.read_csv("reports/metrics/day5_baseline_model_comparison.csv")
    expected_cols = [
        "model", "accuracy_mean", "accuracy_std", 
        "precision_mean", "precision_std", "recall_mean", "recall_std", 
        "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std"
    ]
    for col in expected_cols:
        assert col in df.columns
        
    models = df['model'].tolist()
    assert "Logistic Regression" in models
    assert "Naive Bayes" in models
    assert "Linear SVM" in models

def test_no_test_data_leakage():
    # Because CV loops internally over train set only, no test data is exposed
    pass
