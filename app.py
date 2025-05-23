import os
import re
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
import pickle
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai

# === Gemini 1.5 Flash config ===
genai.configure(api_key="AIzaSyC_0cpKR2V2bmaxqLg8hIViXppvB9WRznc")
model = genai.GenerativeModel("gemini-1.5-flash")

PDF_DIR = "data/pdfs"
EXCEL_DIR = "data/excels"
INDEX_DIR = "data/faiss"
META_FILE = "data/metadata.pkl"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(EXCEL_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

DOMAIN_KEYWORDS = [
    "compliance", "audit", "regulation", "standard", "certificate",
    "policy", "deadline", "inspection", "factory", "environment",
    "quality", "safety", "risk", "report", "law"
]

def extract_entities_regex(text):
    entities = {}
    standards = re.findall(r'(ISO\s?\d{4,6}|IEC\s?\d{4,6}|ANSI\s?\d{4,6})', text, re.IGNORECASE)
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b', text)
    factories = re.findall(r'Factory\sName:\s?([A-Za-z\s]+)', text)
    if standards:
        entities['Standard'] = standards
    if dates:
        entities['Date'] = dates
    if factories:
        entities['Factory'] = factories
    return entities

# === AGENT DEFINITIONS ===
class DocIngestAgent:
    def __init__(self):
        self.metadata_list = []

    def extract_text_from_pdf(self, path):
        with pdfplumber.open(path) as pdf:
            text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:
                    text.append(page_text)
                else:
                    img = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(img)
                    text.append(ocr_text)
            return "\n".join(text)

    def extract_text_from_excel(self, path):
        dfs = pd.read_excel(path, sheet_name=None)
        return "\n\n".join(f"Sheet: {name}\n{df.to_string(index=False)}" for name, df in dfs.items())

    def is_relevant_text(self, text):
        lowered = text.lower()
        return any(kw in lowered for kw in DOMAIN_KEYWORDS)

    def ingest_docs(self, files):
        new_docs = []
        relevant_found = False
        for file in files:
            name = file.name
            ext = name.split(".")[-1].lower()
            dir_path = PDF_DIR if ext == "pdf" else EXCEL_DIR
            path = os.path.join(dir_path, name)

            with open(path, "wb") as f:
                f.write(file.getbuffer())

            raw_text = self.extract_text_from_pdf(path) if ext == "pdf" else self.extract_text_from_excel(path)

            if not self.is_relevant_text(raw_text):
                st.error(f"Uploaded file '{name}' is NOT relevant to compliance domain and will be skipped.")
                continue

            relevant_found = True
            entities = extract_entities_regex(raw_text)
            meta = {"source": name, "entities": entities, "type": "document"}

            for chunk in splitter.split_text(raw_text):
                new_docs.append(Document(page_content=chunk, metadata=meta))
                self.metadata_list.append(meta)

        return new_docs, relevant_found, self.metadata_list

class URLSummarizerAgent:
    def fetch_url_content(self, url):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ComplianceBot/1.0)"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            st.error(f"Failed to fetch URL content: {e}")
            return None

    def extract_main_text_from_html(self, html):
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
            script.decompose()
        texts = soup.stripped_strings
        return "\n".join(texts)

    def is_relevant_text(self, text):
        return any(kw in text.lower() for kw in DOMAIN_KEYWORDS)

    def summarize_text(self, text, url):
        if not self.is_relevant_text(text):
            st.error("URL content does not seem relevant to compliance domain. Skipping indexing.")
            return None, None, None

        prompt = f"""
You are a Compliance Content Summarizer.
Summarize the following web page content with focus on compliance-related points:

[URL] {url}
[Content]
{text}

Return summary in markdown.
"""
        summary = model.generate_content(prompt).text
        meta = {"source": url, "entities": {}, "type": "url"}
        docs = [Document(page_content=chunk, metadata=meta) for chunk in splitter.split_text(summary)]
        return summary, docs, meta

class QueryHandlerAgent:
    def __init__(self, db, metadata_list):
        self.db = db
        self.metadata_list = metadata_list

    def is_relevant_question(self, question):
        return any(kw in question.lower() for kw in DOMAIN_KEYWORDS)

    def handle_query(self, query):
        if not self.is_relevant_question(query):
            return "❌ Your question appears irrelevant to the compliance domain."

        docs_and_scores = self.db.similarity_search_with_score(query, k=3)
        if not docs_and_scores:
            return "⚠️ No relevant documents found for your query."

        top_score = docs_and_scores[0][1]
        if top_score < 0.75:
            return "⚠️ No relevant content found in the uploaded documents."

        docs = [doc for doc, _ in docs_and_scores]
        context = "\n".join(doc.page_content for doc in docs)
        meta_info = "\n".join(f"{doc.metadata['source']} - Entities: {doc.metadata['entities']}" for doc in docs)

        prompt = f"""
You are a Compliance Analysis Agent.
Use this context to extract:
- Extracted Entities (Standard, Date, Factory)
- Compliance Flags (issues)
- Audit Summary (score, deadline)
Return in markdown format.

[Metadata]\n{meta_info}
[Context]\n{context}
[Query]\n{query}
"""
        return model.generate_content(prompt).text

class MainAgent:
    def __init__(self):
        self.doc_agent = DocIngestAgent()
        self.url_agent = URLSummarizerAgent()
        self.db = None
        self.metadata_list = []

    def load_or_create_db(self):
        if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
            self.db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            self.metadata_list = pickle.load(open(META_FILE, "rb")) if os.path.exists(META_FILE) else []

    def update_db(self, new_docs):
        if new_docs:
            if self.db is None:
                self.db = FAISS.from_documents(new_docs, embeddings)
            else:
                self.db.add_documents(new_docs)
            self.db.save_local(INDEX_DIR)
            with open(META_FILE, "wb") as f:
                pickle.dump(self.metadata_list, f)
            return True
        return False

    def run(self):
        st.set_page_config(page_title="Compliance QA + URL Summarizer")
        st.title("🧠 Compliance QA with Document & URL Ingestion")
        self.load_or_create_db()

        uploaded_files = st.file_uploader("Upload Compliance PDFs or Excel files", type=["pdf", "xlsx"], accept_multiple_files=True)
        if uploaded_files:
            new_docs, relevant, meta_list = self.doc_agent.ingest_docs(uploaded_files)
            if relevant:
                self.metadata_list.extend(meta_list)
                if self.update_db(new_docs):
                    st.success("Relevant documents processed and indexed.")
            else:
                st.error("⚠️ None of the uploaded documents contain relevant compliance information.")

        url_input = st.text_input("Or paste a URL to summarize:")
        if url_input:
            html = self.url_agent.fetch_url_content(url_input)
            if html:
                main_text = self.url_agent.extract_main_text_from_html(html)
                if len(main_text) < 200:
                    st.warning("Extracted text is very short, summary may be low quality.")
                summary, docs, meta = self.url_agent.summarize_text(main_text, url_input)
                if summary:
                    st.markdown("### 🔗 URL Summary")
                    st.markdown(summary)
                    self.metadata_list.append(meta)
                    if self.update_db(docs):
                        st.success("URL content summarized and added to the knowledge base.")

        query = st.text_input("Ask a compliance-related question:")
        if query:
            if self.db is None or len(self.metadata_list) == 0:
                st.error("⚠️ Please upload relevant compliance documents or add URL summaries before asking questions.")
            else:
                query_agent = QueryHandlerAgent(self.db, self.metadata_list)
                st.markdown("### 🤖 Response from Compliance Agent")
                response = query_agent.handle_query(query)
                st.markdown(response)

# === Run the app ===
if __name__ == "__main__":
    agent = MainAgent()
    agent.run()
