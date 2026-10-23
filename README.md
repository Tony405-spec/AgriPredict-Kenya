# AgriPredict Kenya

AgriPredict Kenya is a Streamlit crop-yield prediction prototype for Kenyan farming contexts. It uses generated training data, a scikit-learn model, and a local SQLite database for app analytics.

## Quick Start

```bash
git clone https://github.com/Tony405-spec/AgriPredict-Kenya.git
cd AgriPredict-Kenya
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run agripredict_app.py
```

## Runtime Artifacts

Running the app can create local artifacts such as:

- `trained_model.pkl`
- `agripredict_ml.db`
- Python `__pycache__` directories

These are generated files and should not be committed. Recreate them locally by running the app.

## Model Limitations

The current model is trained from generated agricultural data, not a verified production agronomy dataset. Treat predictions as educational estimates and not as final planting, financial, or food-security advice.

## Main Files

- `agripredict_app.py`: Streamlit app and model workflow.
- `database_manager.py`: SQLite persistence helpers.
- `requirements.txt`: Python runtime dependencies.
- `Untitled13.ipynb`: exploratory notebook.
