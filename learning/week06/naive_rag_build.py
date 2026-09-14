import os
import numpy as np
from openai import OpenAI

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"

# Vocareum proxies the OpenAI API under a custom base URL. Set OPENAI_BASE_URL
# in your environment (see setup instructions) to point the SDK at it instead
# of the default https://api.openai.com/v1.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

client = OpenAI(base_url=OPENAI_BASE_URL)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

print("Setup ok. Ready to build RAG.")


documents = [
    # Coffee
    {"id": "coffee_espresso", "text": (
        "Espresso is a concentrated form of coffee made by forcing hot water "
        "under about 9 bars of pressure through finely ground coffee beans. "
        "A single shot is typically 25 to 30 millilitres and takes 25 to 30 "
        "seconds to extract. Espresso forms the base of drinks like the "
        "latte, cappuccino, and americano."
    )},
    {"id": "coffee_beans", "text": (
        "Coffee beans come primarily from two species: Arabica and Robusta. "
        "Arabica accounts for about 60 percent of world production and is "
        "prized for its smoother, more nuanced flavour. Robusta contains "
        "roughly twice as much caffeine and has a stronger, more bitter taste. "
        "Most commercial espresso blends mix the two."
    )},
    {"id": "coffee_brewing", "text": (
        "Pour-over coffee uses a filter cone to drip near-boiling water "
        "through medium-ground coffee. It typically brews for three to four "
        "minutes and produces a clean, light-bodied cup. French press coffee, "
        "in contrast, steeps coarse grounds directly in hot water for four "
        "minutes before pressing, producing a heavier, oil-rich cup."
    )},
    # Tea
    {"id": "tea_green", "text": (
        "Green tea is made from unoxidised leaves of Camellia sinensis. It is "
        "steeped in water at around 70 to 80 degrees Celsius for one to three "
        "minutes. Hotter water or longer steeping produces a bitter, astringent "
        "cup. Green tea is high in an antioxidant called EGCG."
    )},
    {"id": "tea_black", "text": (
        "Black tea comes from fully oxidised Camellia sinensis leaves. It is "
        "brewed with water at or near boiling — 95 to 100 degrees Celsius — "
        "for three to five minutes. Popular varieties include Assam, Darjeeling, "
        "and Ceylon. Black tea typically contains more caffeine than green tea."
    )},
    {"id": "tea_oolong", "text": (
        "Oolong tea is partially oxidised, sitting between green and black tea "
        "in strength and colour. It is brewed at 85 to 95 degrees Celsius for "
        "two to four minutes. Oolong leaves are often rolled and can be re-steeped "
        "several times, with each infusion revealing different flavour notes."
    )},
    # Hot chocolate
    {"id": "chocolate_traditional", "text": (
        "Traditional hot chocolate is made from melted dark chocolate stirred "
        "into hot milk. The ratio is usually 30 to 50 grams of chocolate per "
        "200 millilitres of milk. Whisking prevents the chocolate from settling. "
        "Some recipes add a pinch of chilli or cinnamon for warmth."
    )},
    {"id": "chocolate_powder", "text": (
        "Instant hot chocolate uses cocoa powder mixed with sugar, milk powder, "
        "and stabilisers. Adding hot water dissolves the mix in seconds. It is "
        "cheaper and faster than the traditional method but has a thinner mouthfeel "
        "and less intense chocolate flavour."
    )},
    {"id": "chocolate_history", "text": (
        "Hot chocolate originated with the Maya and Aztec civilisations, who "
        "drank it cold and bitter, spiced with chilli. Europeans encountered "
        "cacao in the 16th century and gradually sweetened the drink and "
        "served it hot. It remained a luxury until industrial cocoa processing "
        "made it affordable in the 19th century."
    )},
    # Milk-based drinks
    {"id": "milk_latte", "text": (
        "A caffè latte is made with one shot of espresso and around 200 "
        "millilitres of steamed milk topped with a thin layer of microfoam. "
        "The ratio is roughly one part espresso to five parts milk. A "
        "cappuccino uses the same espresso base but has equal parts milk and "
        "foam, giving it a lighter, airier texture."
    )},
]

print(f"Loaded {len(documents)} documents.")
for d in documents:
    print(f"  {d['id']:25s}  {len(d['text'])} chars")


def chunk_text(text, size=200, overlap=40):
    """Sliding window over characters."""
    if len(text) <= size:
        return [text]
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = end - overlap
    return chunks

# Chunk every document; build a flat list with source pointer
all_chunks = []
for doc in documents:
    for chunk_idx, chunk in enumerate(chunk_text(doc["text"])):
        all_chunks.append({
            "chunk_id":  f"{doc['id']}#{chunk_idx}",
            "source_id": doc["id"],
            "text":      chunk,
        })

print(f"Total chunks from {len(documents)} documents: {len(all_chunks)}\n")
for c in all_chunks[:8]:  # show first 8 to avoid wall of text
    marker = "…" if len(c["text"]) == 200 else " "
    print(f"  {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
print(f"  ... and {len(all_chunks) - 8} more chunks")


import re

def chunk_by_sentence(text):
    """Split on sentence boundaries. Simpler than a proper NLP splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def chunk_by_paragraph(text):
    """One chunk per paragraph."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

sample = documents[0]["text"]  # coffee_espresso
print(f"Sample document: {documents[0]['id']} ({len(sample)} chars)\n")
print(f"Text: {sample}\n")
print("═" * 80)

sw_chunks = chunk_text(sample)
print(f"\n1. SLIDING WINDOW (size=200, overlap=40) → {len(sw_chunks)} chunks")
for i, c in enumerate(sw_chunks):
    print(f"    [{i}] ({len(c):3d} chars) {c!r}")

sent_chunks = chunk_by_sentence(sample)
print(f"\n2. BY SENTENCE → {len(sent_chunks)} chunks")
for i, c in enumerate(sent_chunks):
    print(f"    [{i}] ({len(c):3d} chars) {c!r}")

para_chunks = chunk_by_paragraph(sample)
print(f"\n3. BY PARAGRAPH → {len(para_chunks)} chunks")
for i, c in enumerate(para_chunks):
    print(f"    [{i}] ({len(c):3d} chars) {c!r}")

print("\n" + "═" * 80)
print("Discussion:")
print("  - Sliding window: uniform sizes but cuts words/sentences")
print("  - Sentence: clean units but variable size (some too small)")
print("  - Paragraph: cleanest semantically but often too large per chunk")
print("  - Real systems combine strategies. We use sliding window for now.")

def embed_batch(texts):
    """One API call, list of 1536-dim vectors back."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]

chunk_texts = [c["text"] for c in all_chunks]
vectors = embed_batch(chunk_texts)

for chunk, vec in zip(all_chunks, vectors):
    chunk["vector"] = vec

print(f"Embedded {len(vectors)} chunks.")
print(f"Each embedding shape: {len(vectors[0])} dimensions")
print(f"\nFirst 8 dims of chunk 0 ({all_chunks[0]['chunk_id']}):")
print(f"  {vectors[0][:8]}")
print(f"\nFirst 8 dims of chunk 1 ({all_chunks[1]['chunk_id']}):")
print(f"  {vectors[1][:8]}")
print("\nTakeaway: embeddings are floats. You cannot read them by eye.")
print("But you CAN compute similarity between them — next cell.")

def cosine(a, b):
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

QUERY = "how is espresso made?"
query_vector = embed_batch([QUERY])[0]

scored = [(cosine(query_vector, c["vector"]), c) for c in all_chunks]
scored.sort(key=lambda pair: pair[0], reverse=True)

print(f"Query: {QUERY!r}\n")
print(f"  {'cosine':>7s}  {'chunk_id':<30s}  preview")
print(f"  {'------':>7s}  {'--------':<30s}  {'-------':<50s}")
for score, chunk in scored[:10]:
    print(f"  {score:>7.3f}  {chunk['chunk_id']:<30s}  {chunk['text'][:50]}...")

def show_top_k(query, k=3):
    """Given a query, print the top-K chunks with their cosine scores."""
    q_vec = embed_batch([query])[0]
    scored = [(cosine(q_vec, c["vector"]), c) for c in all_chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    print(f"\nQ: {query!r}")
    for i, (score, chunk) in enumerate(scored[:k], 1):
        print(f"  [{i}] {score:.3f}  {chunk['chunk_id']:<25s} {chunk['text'][:60]}...")

# Explore different query types:
show_top_k("how much caffeine is in tea?")           # cross-topic (tea + caffeine)
show_top_k("what is a latte?")                        # very specific
show_top_k("traditional recipe")                       # ambiguous (which drink?)
show_top_k("origin of hot chocolate in the 16th century")  # multi-facet
show_top_k("how do you brew beer?")                   # out-of-scope
show_top_k("what is RAG?")

def retrieve(query, chunks, k=3):
    q_vec = embed_batch([query])[0]
    scored = [(cosine(q_vec, c["vector"]), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**c, "score": s} for s, c in scored[:k]]

top3 = retrieve(QUERY, all_chunks, k=3)

print(f"Top 3 chunks for query {QUERY!r}:\n")
for i, hit in enumerate(top3, 1):
    print(f"  [{i}] {hit['chunk_id']}  (cosine {hit['score']:.3f})")
    print(f"      {hit['text']}\n")


SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say so "
    "plainly. Cite the source id in square brackets after any fact you use."
)

def build_prompt(question, retrieved, system=SYSTEM):
    context = "\n\n".join(
        f"[{hit['chunk_id']}]\n{hit['text']}"
        for hit in retrieved
    )
    user_msg = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
    return system, user_msg

system_msg, user_msg = build_prompt(QUERY, top3)

print("═══ SYSTEM ═══")
print(system_msg)
print("\n═══ USER ═══")
print(user_msg)


def ask_rag(question, chunks, k=3, system=SYSTEM):
    retrieved = retrieve(question, chunks, k=k)
    sys, user = build_prompt(question, retrieved, system=system)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user",   "content": user},
        ],
    )
    return {
        "question":   question,
        "answer":     resp.choices[0].message.content,
        "sources":    [hit["chunk_id"] for hit in retrieved],
        "tokens_in":  resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
    }

result = ask_rag(QUERY, all_chunks, k=3)

print(f"Q: {result['question']}\n")
print(f"A: {result['answer']}\n")
print(f"Sources retrieved: {result['sources']}")
print(f"Prompt tokens: {result['tokens_in']}  ·  Completion tokens: {result['tokens_out']}")


SYSTEM_STRICT = (
    "You are a strict assistant. Answer ONLY from the context provided. "
    "If the answer is not explicitly stated in the context, respond with "
    "'The context does not contain this information.' Do not use any "
    "knowledge from your training. Cite the source id after each fact."
)

SYSTEM_PERMISSIVE = (
    "You are a helpful assistant. Use the context to answer, but you may "
    "add helpful background from what you know if it clarifies the answer."
)

SYSTEM_NO_INSTRUCTION = (
    "You are a helpful assistant."
)

test_query = "What is the ideal water temperature for oolong tea, and why?"

for label, sys_prompt in [
    ("STRICT     ", SYSTEM_STRICT),
    ("PERMISSIVE ", SYSTEM_PERMISSIVE),
    ("NO INSTRUCT", SYSTEM_NO_INSTRUCTION),
]:
    r = ask_rag(test_query, all_chunks, k=3, system=sys_prompt)
    print(f"── {label} ──")
    print(f"   {r['answer']}")
    print(f"   sources: {r['sources']}\n")

r = ask_rag("how do you brew beer?", all_chunks, k=3)
print(f"Q: {r['question']}\n")
print(f"A: {r['answer']}\n")
print(f"Sources retrieved: {r['sources']}")
# Discussion:
#   - Cosine still returned SOMETHING. Look at the sources.
#   - Did the LLM refuse (correct)?
#   - Or did it try to answer from training data (violates our grounding rule)?


sys_msg, user_msg = build_prompt("how is espresso made?", [])
print("═══ USER MESSAGE WITH ZERO CONTEXT ═══")
print(user_msg)
print()

resp = client.chat.completions.create(
    model=CHAT_MODEL,
    temperature=0.0,
    messages=[
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ],
)
print("═══ ANSWER ═══")
print(resp.choices[0].message.content)
# Discussion: did the LLM follow the rule (refuse) or leak training knowledge?


r = ask_rag("expresso how mak??", all_chunks, k=3)
print(f"Q: {r['question']}\n")
print(f"A: {r['answer']}\n")
print(f"Sources retrieved: {r['sources']}")
# Discussion: embeddings tolerate typos surprisingly well. Retrieval likely
# still found the coffee chunks. This is embeddings' strength.


r = ask_rag(
    "Compare the caffeine content of black tea and green tea, and the water "
    "temperatures used to brew them.",
    all_chunks, k=3,
)
print(f"Q: {r['question']}\n")
print(f"A: {r['answer']}\n")
print(f"Sources retrieved: {r['sources']}\n")
print("Retrieved chunks (with content):\n")
for hit in retrieve(r['question'], all_chunks, k=3):
    print(f"  [{hit['chunk_id']}] cosine={hit['score']:.3f}")
    print(f"    {hit['text'][:120]}...\n")
# Discussion:
#   - This is a multi-fact question. Does k=3 give us BOTH tea chunks?
#   - If not, the LLM has half the answer. Did it fabricate the other half?
#   - This is a common naive-RAG failure — top-3 doesn't cover both facts.