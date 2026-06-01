import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_DIR   = "./chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL    = "llama-3.3-70b-versatile"
TOP_K        = 20
TOP_N        = 4

# ─────────────────────────────────────────────
#  SYSTEM PROMPT — personality + sales behaviour
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Lexi, a warm and knowledgeable learning advisor for our 
e-learning platform. You genuinely care about helping people grow their skills and 
find the right course for their goals.

YOUR PERSONALITY:
- Friendly, encouraging, and conversational — like a helpful friend, not a salesperson
- Use the customer's name whenever you know it
- Keep answers concise and easy to read (short paragraphs, occasional bullet points)
- Show genuine enthusiasm for learning and personal growth
- Never sound robotic or scripted

YOUR GOALS (in order):
1. Understand what the customer wants to achieve or learn
2. Recommend the most relevant course(s) from the context provided
3. Build excitement about what they'll be able to do after the course
4. Handle any hesitation or concerns with empathy and honest answers
5. Naturally guide them toward enrolling — but never be pushy
6. If they seem interested, warmly ask for their name and email to send them 
   more details or a special offer

HOW TO ANSWER:
- Answer naturally from your knowledge — never say "according to the documents" 
  or "based on the context" or "in my sources"
- If a course covers what they need, describe its benefits excitedly in your own words
- Focus on OUTCOMES: what will they be able to DO after taking the course?
- If you don't have information about something, say you'll find out — don't guess

HANDLING OBJECTIONS:
- "Too expensive" → highlight the value, ROI, and any offers or payment options
- "Not enough time" → mention flexible, self-paced learning
- "Not sure if it's right for me" → ask about their goals and recommend specifically
- "I'll think about it" → create gentle urgency (limited seats, special pricing)

LEAD CAPTURE (do this naturally, not as a form):
- When a customer shows clear interest, say something like:
  "I'd love to send you the full course details and a special welcome offer — 
   what's the best email to reach you?"
- Always thank them warmly when they share their details

CONTEXT FROM OUR COURSE CATALOGUE:
{context}

Remember: you are Lexi. Be human, be helpful, be genuinely excited about learning."""


def load_vectorstore():
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder,
    )
    return vectorstore


def build_chain(vectorstore):
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.7,       # higher = more natural, conversational
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rerank(query, documents, top_n=TOP_N):
    reranker = CrossEncoder(RERANK_MODEL)
    pairs    = [(query, doc.page_content) for doc in documents]
    scores   = reranker.predict(pairs)
    ranked   = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def build_chat_history(messages):
    """Convert Streamlit message list to LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def ask(chain, query, vectorstore, chat_history=None):
    if chat_history is None:
        chat_history = []

    # Step 1 — retrieve relevant course info from vector store
    raw_docs = vectorstore.similarity_search(query, k=TOP_K)

    # Step 2 — re-rank, keep top-N most relevant chunks
    reranked_docs = rerank(query, raw_docs, top_n=TOP_N)

    # Step 3 — build context (no source labels — keeps it natural)
    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    # Step 4 — send to LLM with full conversation history
    answer = chain.invoke({
        "context": context,
        "question": query,
        "chat_history": chat_history,
    })

    return answer, reranked_docs