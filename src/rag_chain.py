import re
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve
from context_compressor import compress_context

load_dotenv()

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model.generation_config.max_length = None
model.generation_config.max_new_tokens = 512

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.15,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

print("Model loaded successfully")

template = """You are a scientific research assistant. Answer the question using ONLY the context below.

Rules:
- Give a clear and concise answer in 2-3 sentences.
- Use exact material names, numerical values, and units from the context.
- Include all relevant numbers, material names, and properties mentioned in the context that are needed to fully answer the question.
- Do NOT repeat sentences.
- If the answer is not in the context, say "I don't have enough information to answer this."
- If the question names a specific dopant combination, report ONLY the value for that exact material. Do NOT list values from other compositions.
- Do NOT combine numerical values from different studies into one answer.
- Report the single most specific value from the context. Do not average or merge values from different sources.
- Prefer information from introduction or results sections over reference lists.
- Do not cite paper references from the reference list as answers.

Example:
Question: What is the d33 of BIT ceramics?
Bad answer: The piezoelectric properties of various ceramics have been studied extensively in the literature.
Good answer: BIT ceramics exhibit a piezoelectric coefficient d33 of 20 pC/N as reported in the results section.

Conversation History:
{history}

<context>
{context}
</context>

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)


def rewrite_query(query, history_text):
    if not history_text.strip():
        return query

    ambiguous_terms = [
        "it", "this", "that", "they", "these",
        "the material", "the ceramic", "the compound",
        "same", "above"
    ]

    query_lower = query.lower()
    needs_rewrite = any(f" {term} " in f" {query_lower} " for term in ambiguous_terms)

    if not needs_rewrite:
        return query

    rewrite_prompt = f"""Given the conversation history below, rewrite the last question to be fully self-contained and clear.
Only return the rewritten question, nothing else.

Conversation History:
{history_text}

Original Question: {query}

Rewritten Question:"""

    inputs = tokenizer(rewrite_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    rewritten = text.strip().split("\n")[0].strip()
    rewritten = rewritten.split("(")[0].strip()
    rewritten = rewritten.split("Note:")[0].strip()

    if len(rewritten) < 10 or len(rewritten) > 200:
        return query

    return rewritten


def clean_answer(answer):
    answer = re.sub(r'</s>', '', answer)
    answer = re.sub(r'<[^>]+>', '', answer)
    answer = re.sub(r'_REF_\S+', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    for stop in ["Note:", "Human:", "Question:", "Context:", "| Human:",
                 "However,", "But let me", "Let me rephrase", "To clarify",
                 "If you meant", "If you could", "Please note"]:
        if stop in answer:
            candidate = answer.split(stop)[0].strip()
            if len(candidate) > 20:
                answer = candidate
                break
    return answer.strip()


def get_answer(query, chat_history=None):
    if chat_history is None:
        chat_history = []

    history_text = "\n".join(chat_history[-6:])

    if not chat_history:
        rewritten_query = query
    else:
        rewritten_query = rewrite_query(query, history_text)

    print(f"[Rewritten Query]: {rewritten_query}")

    chunks = retrieve(rewritten_query)

    if not chunks:
        return "I don't have enough information to answer this.", chat_history

    chunks = compress_context(rewritten_query, chunks)

    context = "\n\n".join(chunk["text"] for chunk in chunks)

    chain = prompt | llm

    result = chain.invoke({
        "history": history_text if history_text else "No previous conversation.",
        "context": context,
        "question": rewritten_query
    })

    answer = clean_answer(result)

    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {answer}")
    chat_history[:] = chat_history[-6:]

    return answer, chat_history


if __name__ == "__main__":
    print("Research Paper Assistant (type 'exit' to quit)\n")

    history = []

    while True:
        query = input("You: ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer, history = get_answer(query, history)
        print(f"\nAssistant: {answer}\n")
        print("-" * 50)