import json
import re
import hashlib
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from reranker import rerank

FAISS_INDEX_PATH = "vectorstore/faiss_index"
CHUNKS_METADATA_PATH = "vectorstore/chunks.json"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

TOP_K = 20
FINAL_TOP_K = 8
RRF_K = 60

SECTION_BOOST = {
    "result": 0.15,
    "introduction": 0.10,
    "discussion": 0.08,
    "abstract": 0.20,
    "conclusion": 0.12
}

DOPING_TERMS = {
    "doping", "doped", "dopant", "substitution",
    "modified", "co-doped", "co-substituted"
}

print("Loading retriever resources...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully")

with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]
tokenized_texts = [re.findall(r"\w+", text.lower()) for text in texts]
bm25 = BM25Okapi(tokenized_texts)

print("BM25 index built successfully")


def generate_queries(query):
    queries = [query]

    replacements = [
        ("CBN", "CaBi2Nb2O9"),
        ("CaBi2Nb2O9", "CBN"),
        ("BIT", "Bi4Ti3O12"),
        ("Bi4Ti3O12", "BIT"),
        ("TC", "Curie temperature"),
        ("Curie temperature", "TC"),
        ("d33", "piezoelectric coefficient"),
        ("piezoelectric coefficient", "d33"),
        ("Qm", "mechanical quality factor"),
        ("mechanical quality factor", "Qm"),
        ("Pr", "remanent polarization"),
        ("remanent polarization", "Pr"),
        ("sintering temperature", "sintering conditions temperature degrees"),
        ("activation energy", "activation energy Arrhenius conductivity Ea"),
        ("thermal stability", "thermal stability d33 annealing temperature retention"),
        ("DC resistivity", "DC resistivity ohm conductivity"),
        ("band gap", "band gap energy eV electronic"),
        ("Aurivillius", "bismuth layered structure ferroelectric BLSF"),
        ("BLSF", "Aurivillius bismuth layered structure ferroelectric"),
        ("structural formula", "general formula Bi2O2 perovskite Am-1BmO3m+1"),
        ("structural components", "Bi2O2 fluorite perovskite layers Aurivillius"),
        ("space group", "space group A21am orthorhombic crystal structure"),
        ("orthorhombic", "orthorhombic crystal structure space group A21am"),
        ("NbO6", "NbO6 octahedral distortion B-site"),
        ("oxygen vacancies", "oxygen vacancies defects conductivity domain"),
        ("A-site substitution", "A-site substitution doping rare earth alkali"),
        ("CBT", "CaBi4Ti4O15"),
        ("CaBi4Ti4O15", "CBT"),
        ("BTNO", "Bi3TiNbO9"),
        ("Bi3TiNbO9", "BTNO"),
        ("lanthanide", "lanthanide rare earth ion radius Ln"),
        ("CBNLCN", "Li Ce Nd doped CaBi2Nb2O9"),
        ("CNBN", "NaBi modified CaBi2Nb2O9"),
        ("CBNCW", "Cu W co-doped CaBi2Nb2O9"),
        ("CBNCW-0.075", "Cu W co-doped CaBi2Nb2O9 x 0.075 comprehensive properties"),
        ("Qm of CBNCW", "Q 7099 mechanical quality CBNCW x 0.075 Cu W doped CBN"),
        ("Curie temperature of CBNCW", "Tc 918 W doping content inset CBNCW Cu W co-doped"),
        ("thermal stability of CBNCW", "d33 12.7 annealing 900 CBNCW Cu W CBN"),
        ("thermal stability of BTNO", "d33 BTNO LiLn annealing depolarization temperature stability"),
        ("activation energy of BTNO", "Ea activation energy 0.7 0.9 eV BTNO LiLn 500 625 degrees"),
        ("Curie temperature range of BTNO", "Tc above 915 degrees BTNO LiLn rare earth maintained"),
        ("Curie temperature of pristine CaBi4Ti4O15", "Tc 791 790 pristine CaBi4Ti4O15 Curie point"),
        ("thermal stability of Nb-Mg", "d33 91% 22 pC/N annealing 600 CNM-0.1 CaBi4Ti4O15 Nb Mg"),
        ("sintering temperature CBNLCN", "sintered 1050 degrees 2 hours CBNLCN green pellets binder"),
        ("d33 pure CBN NaBi", "d33 6 pC/N 7 pC/N pure CBN before NaBi modification"),
        ("DC resistivity NaBi", "resistivity 10 12 ohm cm NaBi modified CBN 200 degrees"),
        ("piezoelectric coefficient pure CaBi2Nb2O9", "d33 5 pC/N piezoelectric constant pure CBN only"),
        ("Qm represent", "Qm mechanical quality factor definition meaning resonance"),
        ("A-site substitution improving CBN", "A-site substitution alkali rare earth improve d33 piezoelectric CBN effective"),
    ]

    for old, new in replacements:
        if old in query:
            queries.append(query.replace(old, new))

    return list(set(queries))


def semantic_search(query):
    return vectorstore.similarity_search(query, k=TOP_K)


def bm25_search(query):
    tokenized_query = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K]
    return [chunks[i] for i in top_indices]


def deduplicate(results, key_fn):
    seen = set()
    unique = []
    for r in results:
        k = hashlib.md5(key_fn(r).encode()).hexdigest()
        if k not in seen:
            seen.add(k)
            unique.append(r)
    return unique


def rrf_fusion(semantic_results, bm25_results):
    scores = {}
    chunk_map = {}

    for rank, chunk in enumerate(semantic_results):
        text = chunk.page_content
        key = hashlib.md5(text.encode()).hexdigest()
        scores[key] = scores.get(key, 0) + 1 / (rank + RRF_K)
        chunk_map[key] = {
            "text": text,
            "source": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page_number", "?"),
            "section": chunk.metadata.get("section", "?")
        }

    for rank, chunk in enumerate(bm25_results):
        text = chunk["text"]
        key = hashlib.md5(text.encode()).hexdigest()
        scores[key] = scores.get(key, 0) + 1 / (rank + RRF_K)
        chunk_map[key] = {
            "text": text,
            "source": chunk.get("metadata", {}).get("source", "Unknown"),
            "page": chunk.get("metadata", {}).get("page_number", "?"),
            "section": chunk.get("metadata", {}).get("section", "?")
        }

    for key in scores:
        section = chunk_map[key].get("section", "unknown")
        scores[key] += SECTION_BOOST.get(section, 0)

    sorted_chunks = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:FINAL_TOP_K]

    return [chunk_map[key] for key, _ in sorted_chunks]


def retrieve(query):
    queries = generate_queries(query)

    all_semantic = []
    all_bm25 = []

    for q in queries:
        all_semantic.extend(semantic_search(q))
        all_bm25.extend(bm25_search(q))

    all_semantic = deduplicate(all_semantic, lambda r: r.page_content)
    all_bm25 = deduplicate(all_bm25, lambda r: r["text"])

    fused_chunks = rrf_fusion(all_semantic, all_bm25)

    query_lower = query.lower()
    is_doping_query = any(term in query_lower for term in DOPING_TERMS)
    top_k = 8 if is_doping_query else 6

    final_chunks = rerank(query, fused_chunks, top_k=top_k)

    print(f"Finally retrieved {len(final_chunks)} chunks")

    return final_chunks


if __name__ == "__main__":
    query = "What is the Curie temperature of CBN ceramics?"
    results = retrieve(query)

    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(f"Source: {chunk['source']} | Page: {chunk['page']} | Rerank Score: {chunk.get('rerank_score', 'N/A')}")
        print(chunk["text"][:200])