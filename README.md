# Context-Aware RAG Chatbot (Groq + Chroma + Memory + Web Search)

This project implements a **context-aware conversational chatbot** using **Retrieval-Augmented Generation (RAG)**.  
The chatbot can:
- Remember conversation history (context awareness)
- Retrieve answers from a custom document corpus
- Optionally search the web for up-to-date information
- Generate grounded, reliable responses using a large language model
- Run as an interactive web app using Streamlit

---

## 🚀 Features

- **Context Memory**
  - Maintains conversation history so follow-up questions are understood correctly.

- **Retrieval-Augmented Generation (RAG)**
  - Custom documents are embedded and stored in a vector database (Chroma).
  - Relevant document chunks are retrieved for every user query.

- **Web Search (Optional)**
  - Toggle web search on/off from the UI.
  - Uses DuckDuckGo search to fetch relevant online snippets.
  - Clearly separates document sources and web sources.

- **LLM Integration (Groq)**
  - Uses Groq-hosted LLaMA / Mixtral / Gemma models.
  - Automatic fallback if a model is deprecated.

- **Streamlit Deployment**
  - Clean chat-style UI.
  - Displays question, answer, and sources explicitly.

---

## 🧠 System Architecture


