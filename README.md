---
title: RecalLens
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---
# RecalLens 🔍

RecalLens is a study companion for university students. Upload your PDF notes, ask questions with page-level citations, and review automatically generated flashcards using spaced repetition!

## Demo Credentials
- **Email:** `demo@recallens.app`
- **Password:** `Demo1234`

## How to Test Locally
1. Navigate to the project folder.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Hugging Face token. Create a `.streamlit/secrets.toml` file with:
   ```toml
   HF_TOKEN = "your_hugging_face_token"
   ```
   *Alternatively, set it as an environment variable (`export HF_TOKEN="your_hugging_face_token"` or `$env:HF_TOKEN="your_hugging_face_token"` in PowerShell).*
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Hugging Face Spaces (Streamlit SDK)
1. Log into your Hugging Face account and create a new **Space**.
2. Select **Streamlit** as the Space SDK.
3. In your space settings, go to **Settings > Variables and secrets** and add a new **Secret**:
   - **Name:** `HF_TOKEN`
   - **Value:** Your Hugging Face access token (ensure it has permissions for Meta-Llama-3.1-8B-Instruct).
4. Upload the following files to your space repository:
   - `app.py`
   - `requirements.txt`
   - `sample_notes.pdf`
5. The space will automatically build and deploy!

## Stack
- Hugging Face Inference API (Meta-Llama-3.1-8B-Instruct with automatic fallback to Qwen2.5-7B)
- ChromaDB for vector storage
- Sentence-Transformers (all-MiniLM-L6-v2) for embeddings
- PyMuPDF for PDF parsing
- Streamlit for the UI
