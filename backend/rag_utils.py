
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
import re

CITY_TO_CODE = {"atlanta": "ASH-ATL", "boston": "ASH-BOS", "chicago": "ASH-CHI",
                "denver": "ASH-DEN", "miami": "ASH-MIA", "seattle": "ASH-SEA"}

# Guest wording -> the policy's own terms (appended to the search text only)
EXPAND = {("noise", "noisy", "loud", "disturb", "party", "music"): " quiet hours disturbance complaint"}

# Topic words that must pull in a specific policy row
KEYWORD_BOOST = {("noise", "noisy", "loud", "quiet", "party", "music"): "Quiet hours"}

def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    q = query.lower()

    codes = set(re.findall(r"ASH-[A-Z]{3}", query)) | {c for city, c in CITY_TO_CODE.items() if city in q}

    search_query = query
    for words, extra in EXPAND.items():
        if any(w in q for w in words):
            search_query += extra

    ranked = vectorstore.similarity_search(search_query, k=vectorstore.index.ntotal)

    # 1) Boost chunks that mention the property named in the question
    if codes:
        n = 2 * len(codes)
        hits = [d for d in ranked if any(c in d.page_content for c in codes)]
        hit_ids = {id(d) for d in hits[:n]}
        ranked = hits[:n] + [d for d in ranked if id(d) not in hit_ids]
        k = min(k, n + 2)        # property row(s) + 2 general chunks

    # 2) Boost the policy row for topic words such as noise or quiet hours
    for words, phrase in KEYWORD_BOOST.items():
        if any(w in q for w in words):
            front = [d for d in ranked if phrase in d.page_content][:1]
            front_ids = {id(d) for d in front}
            ranked = front + [d for d in ranked if id(d) not in front_ids]

    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in ranked[:k]
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""
    You are an AI assistant answering questions about AuroraStay hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {context}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exceptions, fees, limits, or approval requirements.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent fees, dates, amenities, exceptions, or other policy details.
    10. When the question names a property, use only the row for that exact property code. Never use another property's values; if that property's row is missing, say so.
    11. If the retrieved information is missing something the question asks for, say exactly what is missing (name the property or topic asked about) and recommend checking the live property profile or contacting the property team. Still share any related information that is present. If the retrieved information conflicts, clearly mention the conflict.
    12. End with 'Source: <file name>' using only the file name from the metadata. Never output URLs.

    Answer:
    """

    return model.invoke(prompt)


# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
