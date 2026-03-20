import os
import re
import json
import fitz
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

PAPERS_DIR = "data/papers"
FAISS_INDEX_PATH = "vectorstore/faiss_index"
CHUNKS_METADATA_PATH = "vectorstore/chunks.json"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "method",
    "approach",
    "experiment",
    "evaluation",
    "result",
    "discussion",
    "conclusion",
    "future work",
    "reference"
]


def detect_section(text):
    text_lower = text.lower()
    for keyword in SECTION_KEYWORDS:
        if re.search(rf'\b{keyword}\b', text_lower):
            return keyword
    return "unknown"


def is_garbled(text):
    words = text.split()
    if not words:
        return True
    long_merged = sum(1 for w in words if len(w) > 40)
    if long_merged / len(words) > 0.15:
        return True
    if re.match(r'^\s*F\s*I\s*G\s*U\s*R\s*E', text, re.IGNORECASE):
        return True
    if re.match(r'^\s*T\s*A\s*B\s*L\s*E', text, re.IGNORECASE):
        return True
    return False


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(fig\.|figure|table)\s*\d+.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def extract_tables_from_page(page, page_num, source, file_path, section):
    table_texts = []
    try:
        tables = page.find_tables()
        for table in tables:
            try:
                df = table.to_pandas()
                if df.empty:
                    continue

                df = df.fillna("")
                df.columns = [str(c).strip() for c in df.columns]

                rows = []
                header = " | ".join(str(c) for c in df.columns if str(c).strip())
                if header.strip():
                    rows.append(header)

                for _, row in df.iterrows():
                    row_text = " | ".join(
                        str(v).strip() for v in row.values
                        if str(v).strip()
                    )
                    if row_text.strip() and len(row_text.split()) > 1:
                        rows.append(row_text)

                if rows:
                    table_text = "\n".join(rows)
                    table_texts.append({
                        "text": table_text,
                        "page_number": page_num,
                        "section": section,
                        "source": source,
                        "file_path": file_path
                    })
            except Exception:
                continue
    except Exception:
        pass
    return table_texts


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_data = []
    current_section = "unknown"
    paper_title = Path(pdf_path).stem.replace("_", " ").title()

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        if not blocks:
            continue

        page_lines = []

        for block in blocks:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                line_spans = sorted(line["spans"], key=lambda s: s["origin"][0])
                base_size = max(span["size"] for span in line_spans)

                line_text = ""
                for span in line_spans:
                    span_text = span["text"].strip()
                    if not span_text:
                        continue
                    if span["size"] >= base_size * 0.75:
                        if line_text and not line_text.endswith(" "):
                            line_text += " "
                        line_text += span_text

                line_text = line_text.strip()
                if line_text:
                    page_lines.append(line_text)

        if not page_lines:
            continue

        for line in page_lines:
            line_lower = line.strip().lower()
            for keyword in SECTION_KEYWORDS:
                if line_lower == keyword or line_lower.startswith(keyword):
                    current_section = keyword
                    break

        text = clean_text("\n".join(page_lines))

        if not text:
            continue

        pages_data.append({
            "text": text,
            "page_number": page_num,
            "section": current_section,
            "source": paper_title,
            "file_path": Path(pdf_path).name
        })

        table_entries = extract_tables_from_page(
            page, page_num, paper_title,
            Path(pdf_path).name, current_section
        )
        pages_data.extend(table_entries)

    doc.close()
    return pages_data


def load_all_papers(papers_dir):
    all_pages = []
    pdf_files = list(Path(papers_dir).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{papers_dir}'")

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        pages = extract_text_from_pdf(str(pdf_path))
        all_pages.extend(pages)
        print(f"Extracted {len(pages)} pages")

    print(f"Total pages: {len(all_pages)}")
    return all_pages


def pages_to_document(pages_data):
    documents = []

    for page in pages_data:
        doc = Document(
            page_content=page["text"],
            metadata={
                "source": page["source"],
                "page_number": page["page_number"],
                "section": page["section"],
                "file_path": page["file_path"]
            }
        )
        documents.append(doc)

    print(f"Total documents created: {len(documents)}")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    chunks = [c for c in chunks if len(c.page_content.split()) > 20]
    chunks = [c for c in chunks if not is_garbled(c.page_content)]

    for chunk in chunks:
        detected = detect_section(chunk.page_content)
        if detected != "unknown":
            chunk.metadata["section"] = detected

    print(f"Total chunks created: {len(chunks)}")
    return chunks


def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)

    return vectorstore


def save_chunks_for_bm25(chunks):
    os.makedirs(os.path.dirname(CHUNKS_METADATA_PATH), exist_ok=True)

    data = []
    for chunk in chunks:
        data.append({
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Chunks saved to '{CHUNKS_METADATA_PATH}'")


def run_ingestion():
    print("Starting Ingestion Pipeline")

    pages_data = load_all_papers(PAPERS_DIR)
    documents = pages_to_document(pages_data)
    chunks = chunk_documents(documents)
    embed_and_store(chunks)
    save_chunks_for_bm25(chunks)

    print("Ingestion Complete!")


if __name__ == "__main__":
    run_ingestion()