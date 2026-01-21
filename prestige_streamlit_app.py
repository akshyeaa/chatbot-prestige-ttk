# prestige_streamlit_app.py
# Streamlit AI Product Advisor – Production‑style UI + Memory + Multilingual + RAG

import streamlit as st
import uuid
import re
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ================= ENV SETUP =================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ================= STREAMLIT PAGE =================
st.set_page_config(page_title="Prestige AI Advisor", layout="wide")

# ================= FIXED HEADER STYLE =================
st.markdown(
    """
    <style>
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: white;
        z-index: 1000;
        padding: 10px 20px;
        border-bottom: 1px solid #eee;
    }
    .chat-area {
        margin-top: 120px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================= SESSION SETUP =================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= GLOBAL MEMORY STORE (IMPORTANT FIX) =================
if "memory_store" not in st.session_state:
    st.session_state.memory_store = {}

# ================= HEADER =================
st.markdown("<div class='fixed-header'>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("## 🍳 Prestige Smart Product Advisor")
    st.caption("AI‑powered assistant for product search, support & recommendations")

with col2:
    language = st.selectbox("🌐 Language", ["English", "Hindi", "Kannada"], index=0)

st.markdown("</div>", unsafe_allow_html=True)

# ================= VECTOR DB (SCRAPE ONCE & CACHE) =================
@st.cache_resource(show_spinner="🔎 Loading Prestige product catalog...")
def load_vector_db():

    urls = []
    categories = {
        "gas stove": "https://shop.ttkprestige.com/catalogsearch/result/index/?q=gas+stove&p=",
        "pressure cooker": "https://shop.ttkprestige.com/catalogsearch/result/index/?q=pressure+cooker&p=",
        "induction": "https://shop.ttkprestige.com/catalogsearch/result/index/?q=induction&p=",
        "cookware": "https://shop.ttkprestige.com/catalogsearch/result/index/?q=cookware&p=",
        "mixer grinder": "https://shop.ttkprestige.com/catalogsearch/result/index/?q=mixer+grinder&p=",
        "air fryer": "https://shop.ttkprestige.com/catalogsearch/result/index/?q=air+frier&p="
    }

    for base_url in categories.values():
        for page in range(1, 4):
            urls.append(base_url + str(page))

    loader = WebBaseLoader(urls)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    return db

# Load once
db = load_vector_db()
retriever = db.as_retriever(search_kwargs={"k": 6})

# ================= LLM =================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.25)

# ================= PROMPT (PRODUCT-FIRST, NOT CHAT-FIRST) =================
prompt = ChatPromptTemplate.from_template("""
You are an AI Product Advisor for Prestige Kitchen Appliances.

Language: {language}

Your goal:
- Help users quickly find suitable kitchen products.
- Recommend best matching products based on user needs.

Rules:
- Use only Prestige products from the provided context.
- Focus on product discovery, not casual conversation.
- If product is not available, clearly say so.

Tasks:
- Recommend top 2–3 suitable products.
- For each product include:
  • Product name  
  • Approximate price  
  • Short reason for recommendation  
  • Official Prestige search link for the product  

IMPORTANT:
- Do NOT invent direct product page URLs.
- Always provide links in this format:
  https://shop.ttkprestige.com/catalogsearch/result/index/?q=<product name>

If user has complaints or service issues:
- Provide:
  Service: https://www.ttkprestige.com/service  
  Contact: https://www.ttkprestige.com/contact-us  

Context:
{context}

Conversation History:
{history}

User Query:
{question}

Answer:
""")


# ================= MEMORY ENGINE (FIXED) =================
def get_session_history(session_id: str):
    store = st.session_state.memory_store
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ================= RAG PIPELINE =================
chain = (
    {
        "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
        "question": RunnableLambda(lambda x: x["question"]),
        "language": RunnableLambda(lambda x: x["language"]),
        "history": RunnableLambda(lambda x: x["history"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)

chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# ================= CHAT AREA =================
st.markdown("<div class='chat-area'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Search for products, ask for help, or report an issue...")

# ================= HELPERS =================
def extract_products(text):
    products = []
    pattern = r"(https?://[^\s]+)"
    links = re.findall(pattern, text)

    lines = text.split("\n")
    for line in lines:
        if "₹" in line:
            price_match = re.search(r"₹\s?\d+[\,\d]*", line)
            price = price_match.group() if price_match else "Price not found"
            name = line.replace(price, "").strip()
            link = links.pop(0) if links else None

            products.append({
                "name": name[:90],
                "price": price,
                "link": link
            })
    return products


def generate_followups(answer_text):
    if "stove" in answer_text.lower():
        return [
            "Best stove under 3000",
            "Glass top vs stainless steel stove",
            "Show premium stoves"
        ]
    if "cooker" in answer_text.lower():
        return [
            "Fastest pressure cooker",
            "Induction compatible cooker",
            "Cooker for 4 people"
        ]
    return [
        "Any current offers?",
        "Compare premium products",
        "Show top rated items"
    ]

# ================= WHEN USER ASKS =================
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = chat_chain.invoke(
        {
            "question": user_input,
            "language": language
        },
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

        # ---------- PRODUCT CARDS ----------
        products = extract_products(answer)

        if products:
            st.subheader("🛒 Recommended Products")
            cols = st.columns(3)

            for i, product in enumerate(products[:3]):
                with cols[i % 3]:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; padding:16px; border-radius:14px; box-shadow:2px 2px 10px rgba(0,0,0,0.05)">
                        <h4>{product['name']}</h4>
                        <p><b>Price:</b> {product['price']}</p>
                        <a href="{product['link']}" target="_blank">🔗 View Official Product</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ---------- FOLLOW-UP SUGGESTIONS ----------
        followups = generate_followups(answer)

        if followups:
            st.subheader("💡 Suggested next questions")
            cols = st.columns(len(followups))
            for i, q in enumerate(followups):
                if cols[i].button(q):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

