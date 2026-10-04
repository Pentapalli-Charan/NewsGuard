import os

dirs = [
    "data/raw", "data/interim", "data/processed",
    "notebooks",
    "src/data", "src/preprocessing", "src/features", "src/models", "src/evaluation", "src/explainability", "src/api", "src/utils",
    "models", "artifacts",
    "reports/figures/day1", "reports/metrics", "reports/experiments",
    "tests", "app", "configs"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

files = [
    "src/__init__.py", "src/data/__init__.py", "src/preprocessing/__init__.py", "src/features/__init__.py",
    "src/models/__init__.py", "src/evaluation/__init__.py", "src/explainability/__init__.py", "src/api/__init__.py", "src/utils/__init__.py",
    "requirements.txt", "pytest.ini", "README.md", "MODEL_CARD.md", ".gitignore"
]

for f in files:
    open(f, 'a').close()
