from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("BAAI/bge-reranker-large")


def rerank(query, chunks, top_k=6, threshold=-5.0):
    if not chunks:
        return []

    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = reranker_model.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    reranked_chunks = []
    for chunk, score in ranked[:top_k]:
        if score > threshold:
            chunk["rerank_score"] = round(float(score), 4)
            reranked_chunks.append(chunk)

    return reranked_chunks