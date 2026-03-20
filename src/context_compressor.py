import re

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and",
    "or", "for", "was", "are", "be", "been", "that", "this",
    "with", "as", "at", "by", "on", "not", "have", "has"
}

VALUE_PATTERN = re.compile(
    r'\d+\.?\d*\s*(?:pC/N|pC\s*N-?1|°C|◦C|degrees\s*C|eV|MHz|kHz|kV|'
    r'Ω\s*cm|Ω·cm|ohm\s*cm|μC/cm2|μC/cm²|uC/cm2|%|ppm)',
    re.IGNORECASE
)

KEY_TERMS = re.compile(
    r'\b('
    r'TC|Tc|Curie|d33|piezoelectric|coefficient|'
    r'polarization|remanent|Pr|'
    r'resistivity|dielectric|permittivity|'
    r'ferroelectric|domain|grain|'
    r'sintering|density|'
    r'BLSF|Aurivillius|perovskite|'
    r'A-site|B-site|doping|substitution|'
    r'Qm|quality|factor|activation|energy|'
    r'thermal|stability|annealing|retention|'
    r'band|gap|eV|conductivity|oxygen|vacancy'
    r')\b',
    re.IGNORECASE
)

MATERIAL_PATTERN = re.compile(
    r'\b(?:[A-Z][a-z]?\d*){2,}\b'
)

VALUE_WEIGHT = 2
KEY_TERM_WEIGHT = 1.5
KEYWORD_WEIGHT = 2.0

USE_COMPRESSION = True


NO_COMPRESS_TERMS = [
    "qm", "sintering", "resistivity", "activation energy",
    "pristine", "thermal stability", "a-site", "represent",
    "dc resistivity", "curie temperature of cbncw",
    "curie temperature of pristine",
    "orthorhombic distortion", "orthorhombicity",
    "structural formula", "general formula",
    "value of m", "type of material",
    "structural components", "highest known",
    "advantages", "over pzt", "blsf over",
    "curie temperature range", "btno", "range of btno"
]


def compress_context(query, chunks):
    if not USE_COMPRESSION:
        return chunks

    query_lower = query.lower()
    if any(term in query_lower for term in NO_COMPRESS_TERMS):
        return chunks

    query_words = {
        w.lower() for w in re.findall(r"\w+", query)
        if w.lower() not in STOPWORDS
    }

    compressed_chunks = []

    for chunk in chunks:
        text = chunk["text"]

        # Split on sentence boundaries AND newlines to handle table rows
        raw_splits = re.split(r'(?<=[.!?])\s+', text.strip())

        sentences = [
            s.strip()
            for s in raw_splits
            if len(s.strip().split()) > 4
        ]

        if not sentences:
            compressed_chunks.append(chunk)
            continue

        scored = []

        for sentence in sentences:
            words = {
                w.lower() for w in re.findall(r"\w+", sentence)
                if w.lower() not in STOPWORDS
            }

            keyword_score = (
                len(query_words & words) / max(len(query_words), 1)
            ) * KEYWORD_WEIGHT

            value_score = (
                len(VALUE_PATTERN.findall(sentence)) * VALUE_WEIGHT +
                len(KEY_TERMS.findall(sentence)) * KEY_TERM_WEIGHT +
                len(MATERIAL_PATTERN.findall(sentence)) * VALUE_WEIGHT
            )

            scored.append((sentence, keyword_score + value_score))

        top_indices = sorted(
            range(len(scored)), key=lambda i: scored[i][1], reverse=True
        )[:7]

        for i, sentence in enumerate(sentences):
            if VALUE_PATTERN.search(sentence) and i not in top_indices:
                top_indices.append(i)
                break

        top_indices = sorted(set(top_indices))[:8]

        compressed_text = " ".join(sentences[i] for i in top_indices)

        compressed_chunks.append({
            "text": compressed_text,
            "source": chunk["source"],
            "page": chunk["page"],
            "section": chunk["section"],
            "rerank_score": chunk.get("rerank_score", None)
        })

    if not compressed_chunks:
        return chunks

    return compressed_chunks