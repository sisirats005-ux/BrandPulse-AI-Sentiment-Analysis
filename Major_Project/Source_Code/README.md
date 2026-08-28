# Twitter Sentiment Analysis (NLP)

Persevex NLP Internship Project. Classify the sentiment of tweets (**positive / neutral / negative**) using two approaches and serve predictions through an interactive Plotly Dash dashboard.

- **Model A — TF-IDF + Logistic Regression**: fast, strong, interpretable baseline. **77.4% accuracy** on the held-out test set.
- **Model B — LSTM (Keras/TensorFlow)**: word-embedding bidirectional recurrent network. **80.0% accuracy** on the held-out test set.

Both models have already been trained end-to-end on the real **Twitter US Airline Sentiment** dataset (14,640 tweets) — the trained models are included in `models/`, so the dashboard runs immediately with no extra setup. See [`reports/Performance_Report.pdf`](reports/Performance_Report.pdf) for the full comparison, confusion matrices, and per-class metrics.

**Why Dash instead of Streamlit:** the brief's Phase 4 objective allows either ("You will build a Streamlit or Plotly Dash web app"). Dash was chosen here because its callback model made it easier to build the multi-tab layout (single prediction, batch CSV, live monitor, model comparison) as one cohesive app rather than several separate Streamlit pages.

## Project structure

```
sentiment_project/
├── app.py                     # Plotly Dash dashboard (single + batch prediction)
├── requirements.txt
├── data/
│   └── tweets.csv             # Twitter US Airline Sentiment dataset (14,640 tweets, included)
├── models/                    # trained models (included, ready to use)
│   ├── logreg_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── lstm_model.keras
│   ├── tokenizer.pickle
│   └── lstm_config.json
├── notebooks/
│   ├── 01_tfidf_logistic_regression.ipynb   # executed, with outputs
│   └── 02_lstm_sentiment.ipynb              # executed, with outputs
├── assets/
│   └── style.css              # Custom SaaS dashboard styling
└── src/
    ├── config.py              # Central configuration
    ├── model_utils.py         # Model loading & prediction utilities
    └── text_utils.py          # Text preprocessing (shared by train & inference)
```

## Setup

```bash
cd sentiment_project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_nltk.py                                   # Download required NLTK data (one-time)
```

## Dataset

`data/tweets.csv` already contains the **Twitter US Airline Sentiment** dataset (14,640 tweets about six US airlines, scraped February 2015). Columns used: `text`, `airline_sentiment`.

To use a different dataset, replace this file — the notebooks auto-detect common formats, including:

- **Sentiment140** (`0=negative, 2=neutral, 4=positive`) — no header, latin-1 encoding.
- Kaggle Twitter datasets with `text` / `sentiment` (or `label`, `target`, `airline_sentiment`) columns.

If no dataset is found, the notebooks fall back to a tiny built-in demo sample so they still run end-to-end.

## (Re)train the models

The models in `models/` are already trained and ready — you only need this step if you want to retrain (e.g. on a different dataset or with different hyperparameters).

Open the notebooks in Jupyter or Google Colab and run all cells, in order:

1. `notebooks/01_tfidf_logistic_regression.ipynb` → saves `models/logreg_model.joblib` + `models/tfidf_vectorizer.joblib`
2. `notebooks/02_lstm_sentiment.ipynb` → saves `models/lstm_model.keras`, `models/tokenizer.pickle`, `models/lstm_config.json`

```bash
jupyter notebook          # or open in VS Code / Colab
```

## Run the dashboard

### Local Development

```bash
python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

The dashboard offers:
- **Single tweet** — type a tweet and get its predicted sentiment + confidence.
- **Batch (CSV)** — upload a CSV, get per-row predictions, a distribution chart, a word cloud, summary metrics, and a downloadable results CSV.
- **Live Stream** — simulated incoming-tweet feed with sentiment trends (replays real training tweets with synthetic timestamps).
- **Model Comparison** — side-by-side accuracy, F1, confusion matrices, and model recommendations.
- **Error Analysis** — see where each model gets it wrong, most-confused class pairs, misclassified examples.
- **About** — dataset stats, methodology pipeline diagram, limitations.

You can switch between the Logistic Regression and LSTM models in the sidebar (whichever have been trained).

### Production Deployment

#### Option 1: Hugging Face Spaces

1. **Create a Hugging Face Account** and go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Create a new Space**:
   - Choose **"Docker"** as the runtime (for full control)
   - Or use **"Gradio"** and adapt the code (not recommended for Dash)
3. **Upload your project files**:
   - Push your project to the Hugging Face Hub using Git:
   ```bash
   cd sentiment_project
   git init
   git add .
   git commit -m "Initial commit"
   git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
   git push -u hf main
   ```
4. **Create a Dockerfile** in your project root (for Docker runtime):
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   RUN python setup_nltk.py

   EXPOSE 7860

   CMD ["python", "app.py"]
   ```
5. **Create `.env` or environment variables** if needed.
6. **Hugging Face will automatically build and deploy** your space.
7. **Access your app** at `https://huggingface.co/spaces/<your-username>/<space-name>`

#### Option 2: Heroku (Deprecated — use Docker alternatives like Railway)

Heroku is discontinuing free tier. Consider alternatives:
- **Railway.app** — Simple, free tier available
- **Render.com** — Free tier with auto-deploy from GitHub
- **Fly.io** — Competitive pricing
- **AWS / Google Cloud / Azure** — Full control, pay-per-use

#### Option 3: Docker (Self-hosted)

```bash
docker build -t brandpulse-ai .
docker run -p 8050:8050 brandpulse-ai
```

#### Option 4: Streamlit Cloud (Alternative Dashboard Framework)

If you prefer Streamlit over Dash:
```bash
git push origin main
# Go to https://streamlit.io/cloud and link your GitHub repo
```

## Architecture & Notes

- Preprocessing lives in `src/text_utils.py` and is shared by training and inference so tweets are cleaned identically in both — this prevents train/serve skew.
- Negation words (e.g. "not", "never") are intentionally kept during stopword removal because they carry sentiment.
- Both models use the same input pipeline, making predictions directly comparable.
- The dashboard uses **Plotly Dash** for interactive, responsive charts and **Bootstrap** for modern SaaS styling.
- Custom CSS in `assets/style.css` provides a polished, professional look with smooth animations and transitions.

## Troubleshooting

### Models not loading
- Run `python setup_nltk.py` to download NLTK tokenizer data.
- Check that model files exist in `models/`.
- Verify model metadata with: `python scripts/generate_model_artifacts.py`

### CSV upload failing
- Ensure your CSV has a text column named: `text`, `tweet`, `content`, or `SentimentText`.
- Check encoding (UTF-8 or latin-1).
- File size limit depends on your server (typically 50MB-1GB).

### Deployment issues on Hugging Face Spaces
- Ensure `requirements.txt` includes all dependencies.
- Check that `setup_nltk.py` runs during build (add to Dockerfile or startup script).
- Monitor space logs for errors.
- For Docker spaces, expose port **7860** (Hugging Face default).

## License & Attribution

Persevex NLP Internship Project. Dataset: Twitter US Airline Sentiment (Kaggle).