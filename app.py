import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from duckduckgo_search import DDGS

# -----------------------------
# Load env + Config
# -----------------------------
load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "rag_corpus"

st.set_page_config(page_title="RAG Chatbot (Groq)", page_icon="💬", layout="wide")
st.title("💬 Context-Aware RAG Chatbot (Groq + Chroma + Memory + Web Search)")

# -----------------------------
# Helpers
# -----------------------------
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    return db.as_retriever(search_kwargs={"k": 4})

def get_llm():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error("Missing GROQ_API_KEY. Add it to a .env file in your project root.")
        st.stop()

    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    last_err = None
    for model_name in candidate_models:
        try:
            llm = ChatGroq(model=model_name, temperature=0.2, max_tokens=800)
            llm.invoke("ping")
            st.sidebar.success(f"Using Groq model: {model_name}")
            return llm
        except Exception as e:
            last_err = e

    st.error("All Groq models failed. Last error:\n\n" + str(last_err))
    st.stop()

def format_docs(docs):
    if not docs:
        return "", []

    blocks = []
    sources = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"{src}" + (f" (p.{page})" if page is not None else "")
        blocks.append(f"[DOC SOURCE: {tag}]\n{d.page_content}")
        sources.append(d.metadata)
    return "\n\n---\n\n".join(blocks), sources

def web_search(query: str, k: int = 5):
    """
    Returns a list of dicts: {title, url, snippet}
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        return [], str(e)

    return results, None

def format_web_results(web_results):
    """
    Convert web results into a context block for the LLM.
    """
    if not web_results:
        return ""

    lines = []
    for i, r in enumerate(web_results, start=1):
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()
        snippet = r.get("snippet", "").strip()
        lines.append(f"[WEB {i}] {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n---\n\n".join(lines)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")
enable_web = st.sidebar.checkbox("Enable Web Search", value=False)
web_k = st.sidebar.slider("Web results", min_value=3, max_value=10, value=5, disabled=not enable_web)

# -----------------------------
# Memory (chat history)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# -----------------------------
# Build components
# -----------------------------
retriever = get_retriever()
llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "RULES:\n"
     "1) Prefer the provided DOCUMENT context.\n"
     "2) If Web context is provided, you may use it, but be cautious.\n"
     "3) If the answer is not supported by the context, say you don't know.\n"
     "4) When using web info, mention it is from web results.\n"
     "Keep answers concise and factual."),
    MessagesPlaceholder("chat_history"),
    ("human",
     "Question: {question}\n\n"
     "DOCUMENT CONTEXT:\n{doc_context}\n\n"
     "WEB CONTEXT (optional):\n{web_context}")
])

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask something from your documents (optionally web)...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving docs..."):
            docs = retriever.invoke(user_input)
            doc_context, doc_sources = format_docs(docs)

        web_results = []
        web_error = None
        web_context = ""

        if enable_web:
            with st.spinner("Searching the web..."):
                web_results, web_error = web_search(user_input, k=web_k)
                web_context = format_web_results(web_results)

        with st.spinner("Generating answer..."):
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "chat_history": st.session_state.messages[:-1],
                "question": user_input,
                "doc_context": doc_context,
                "web_context": web_context,
            })

        # Show question explicitly
        st.markdown(f"**Question:** {user_input}")
        st.markdown(f"**Answer:**\n\n{answer}")

        # Sources used
        with st.expander("Sources used"):
            st.subheader("Document sources")
            if doc_sources:
                for s in doc_sources:
                    st.write(s)
            else:
                st.write("No documents retrieved.")

            st.subheader("Web sources")
            if enable_web:
                if web_error:
                    st.write(f"Web search error: {web_error}")
                elif web_results:
                    for r in web_results:
                        st.markdown(f"- **{r['title']}**\n  - {r['url']}\n  - {r['snippet']}")
                else:
                    st.write("No web results retrieved.")
            else:
                st.write("Web search disabled.")

    st.session_state.messages.append(AIMessage(content=answer))
