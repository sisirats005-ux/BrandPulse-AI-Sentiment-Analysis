"""
One-time setup script: downloads the NLTK resources this project needs
(stopwords, wordnet, omw-1.4).

Run this once before starting the dashboard or the notebooks:

    python setup_nltk.py

After this has been run once, src/text_utils.py will find the resources
already cached locally and will not attempt any network access on import.
"""

from src.text_utils import ensure_nltk_resources

if __name__ == "__main__":
    print("Downloading required NLTK resources (stopwords, wordnet, omw-1.4)...")
    ensure_nltk_resources(auto_download=True)
    print("Done. You can now run: streamlit run app.py")
