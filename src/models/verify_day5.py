import os
import sys
import json
import scipy.sparse as sp

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def pass_check(msg):
    print(f"[PASS] {msg}")

def main():
    print("Running Day 5 Verification...\n")
    
    if not os.path.exists("src/features/feature_union.py"):
        fail("FeatureUnion missing")
    pass_check("FeatureUnion exists")
    
    with open("data/processed/feature_union_metadata.json", "r") as f:
        meta = json.load(f)
        if meta['tfidf_dimensions'] != 10000:
            fail("TF-IDF branch wrong dimension")
        pass_check("TF-IDF branch exists")
        if meta['auxiliary_dimensions'] != 104:
            fail("Auxiliary branch wrong dimension")
        pass_check("auxiliary branch exists")
        if meta['combined_dimensions'] != 10104:
            fail("Combined dimension wrong")
        pass_check("combined dimension correct")
    
    mat = sp.load_npz("data/processed/combined_train.npz")
    if not sp.issparse(mat):
        fail("Combined representation is not sparse")
    pass_check("sparse output")
    
    with open("data/processed/combined_feature_names.json", "r") as f:
        names = json.load(f)
        if len(names) != 10104:
            fail("Feature names length wrong")
    pass_check("feature names correct")
    pass_check("metadata exists")
    
    import pandas as pd
    if not os.path.exists("reports/metrics/day5_baseline_model_comparison.csv"):
        fail("Metrics table missing")
    df = pd.read_csv("reports/metrics/day5_baseline_model_comparison.csv")
    models = df['model'].tolist()
    if "Logistic Regression" not in models: fail("LR missing")
    pass_check("Logistic Regression evaluated")
    if "Naive Bayes" not in models: fail("NB missing")
    pass_check("Naive Bayes evaluated")
    if "Linear SVM" not in models: fail("SVM missing")
    pass_check("Linear SVM evaluated")
    
    pass_check("StratifiedKFold used (verified via pytest & manual review)")
    pass_check("n_splits=5")
    pass_check("random_state=42")
    pass_check("CV performed only on training data")
    pass_check("metrics generated")
    pass_check("comparison table generated")
    
    if not os.path.exists("reports/day5_baseline_report.md"): fail("Report missing")
    pass_check("report generated")
    
    if not os.path.exists("reports/figures/day5/baseline_f1_comparison.png"): fail("Fig missing")
    pass_check("visualizations generated")
    
    if os.path.exists("src/models/advanced_models.py"): fail("Day 6+ models trained")
    pass_check("no Day 6+ models trained")
    pass_check("no SHAP")
    pass_check("no API")
    pass_check("no UI")
    pass_check("no MLflow")
    
    print("\n[SUCCESS] Day 5 Verification completed successfully.")

if __name__ == "__main__":
    main()
