# Mini RAG PDF Assistant

A local PDF question-answering app built with FastAPI, Streamlit, LangChain, Gemini, PyPDF, and FAISS. The backend extracts PDF text, splits it into overlapping chunks, embeds the chunks, and retrieves the three most similar chunks for each question. Gemini is instructed to answer only from those retrieved chunks and to say when they do not contain enough information.

## Requirements

- Python 3.10–3.13 (Python 3.12 is recommended; FAISS wheels may not be available for newer Python versions)
- A Google Gemini API key with access to the configured chat and embedding models

## Setup on Windows

Open PowerShell in this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace `your_google_gemini_api_key` with your API key. Keep this file private; it is excluded from Git.

The app loads `.env` automatically. In the same activated PowerShell session, start the FastAPI backend:

```powershell
python -m uvicorn backend.main:app --reload
```

Open a second PowerShell window in the project folder, activate the environment, and start Streamlit:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py
```

Streamlit opens the UI at `http://localhost:8501`. The API health check is at `http://localhost:8000/health`. If PowerShell blocks virtual-environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window and activate again.

## Using the app

1. Select one or more PDF files and choose **Process PDFs**.
2. Wait while text is extracted and embeddings are created.
3. Enter a question and choose **Get answer**.
4. Expand **Relevant source chunks** to inspect the retrieved passages and their file/page locations.

Uploaded PDFs and their FAISS indexes stay in backend memory for the lifetime of the backend process. Restarting the backend clears them. Each browser session gets its own in-memory index; newly uploaded PDFs are added to that session's existing index. Scanned/image-only PDFs need OCR before they can be searched; OCR is not included.

## Configuration

- `GOOGLE_API_KEY` — required for Gemini embeddings and answers.
- `RAG_API_URL` — optional Streamlit setting for the FastAPI base URL; defaults to `http://localhost:8000`.
- Models, chunk size/overlap, and retrieval count are defined near the top of `backend/rag_service.py`.
