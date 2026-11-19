import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Mini RAG PDF Assistant", page_icon="📄", layout="centered")
st.title("📄 Mini RAG PDF Assistant")
st.caption("Upload PDFs and ask questions answered from their contents.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def show_api_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"API error ({response.status_code}): {detail}")


with st.container(border=True):
    st.subheader("1. Add PDF documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are processed in memory and added to this browser session.",
    )
    if st.button("Process PDFs", type="primary", disabled=not uploaded_files):
        files = [
            ("files", (file.name, file.getvalue(), "application/pdf"))
            for file in uploaded_files
        ]
        try:
            with st.spinner("Extracting text and creating searchable embeddings..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    data={"session_id": st.session_state.session_id},
                    files=files,
                    timeout=300,
                )
            if response.ok:
                result = response.json()
                st.session_state.uploaded_files.extend(result["files"])
                st.success(
                    f"Added {len(result['files'])} PDF(s), "
                    f"{result['pages_extracted']} text page(s), and "
                    f"{result['chunks_added']} searchable chunk(s)."
                )
            else:
                show_api_error(response)
        except requests.RequestException as exc:
            st.error(f"Could not reach the FastAPI backend at {API_URL}: {exc}")

    if st.session_state.uploaded_files:
        st.caption("Available documents: " + ", ".join(st.session_state.uploaded_files))

with st.container(border=True):
    st.subheader("2. Ask a question")
    question = st.text_area(
        "Your question",
        placeholder="What are the main points in these documents?",
        max_chars=4000,
    )
    if st.button(
        "Get answer",
        disabled=not st.session_state.uploaded_files or not question.strip(),
    ):
        try:
            with st.spinner("Searching the PDFs and generating an answer..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": question.strip(),
                    },
                    timeout=120,
                )
            if response.ok:
                result = response.json()
                st.markdown("#### Answer")
                st.write(result["answer"])
                with st.expander(f"Relevant source chunks ({len(result['sources'])})"):
                    for index, source in enumerate(result["sources"], start=1):
                        st.markdown(
                            f"**{index}. {source['source']} — page {source['page']}**"
                        )
                        st.write(source["content"])
                        if index < len(result["sources"]):
                            st.divider()
            else:
                show_api_error(response)
        except requests.RequestException as exc:
            st.error(f"Could not reach the FastAPI backend at {API_URL}: {exc}")

st.caption("Answers are grounded in the top three retrieved text chunks.")
