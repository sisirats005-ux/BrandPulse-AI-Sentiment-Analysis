FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Set environment variables
ENV DASH_HOT_RELOAD=False
ENV PYTHONUNBUFFERED=1

# Run the Dash app directly on port 7860
CMD ["python", "-c", "import app; app.app.run_server(host='0.0.0.0', port=7860, debug=False)"]
