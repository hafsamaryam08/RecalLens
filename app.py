import streamlit as st
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import json
import datetime
import os
import uuid

st.set_page_config(page_title="RecalLens", page_icon="🔍", layout="wide")

# -- CUSTOM CSS TO REDUCE PADDING --
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# -- CONSTANTS & INIT --
DEMO_EMAIL = "demo@recallens.app"
DEMO_PASS = "Demo1234"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
LLM_MODELS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-72B-Instruct"
]

def get_hf_token():
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.EphemeralClient()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "collection" not in st.session_state:
    collection_name = f"notes_{st.session_state.session_id}"
    st.session_state.collection = st.session_state.chroma_client.create_collection(collection_name)
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = ""

if "users" not in st.session_state:
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            st.session_state.users = json.load(f)
    else:
        st.session_state.users = {DEMO_EMAIL: DEMO_PASS}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -- AUTH --
if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center;'>🔍 RecalLens</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Your AI-Powered Study Companion</h4>", unsafe_allow_html=True)
    st.write("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])
            
            with tab_login:
                st.subheader("Welcome Back")
                email = st.text_input("Email address", key="login_email")
                password = st.text_input("Password", type="password", key="login_pass")
                
                if st.button("Sign In", use_container_width=True, type="primary"):
                    if email in st.session_state.users and st.session_state.users[email] == password:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Incorrect email or password.")
                        
            with tab_signup:
                st.subheader("Create an Account")
                new_email = st.text_input("Email address", key="signup_email")
                new_password = st.text_input("Password", type="password", key="signup_pass")
                confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
                
                if st.button("Sign Up", use_container_width=True):
                    if not new_email or not new_password:
                        st.error("Please fill in all fields.")
                    elif new_email in st.session_state.users:
                        st.error("An account with this email already exists.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        st.session_state.users[new_email] = new_password
                        with open("users.json", "w") as f:
                            json.dump(st.session_state.users, f)
                        st.success("Account created successfully! Please switch to the Sign In tab.")
    st.stop()

# -- FUNCTIONS --
def extract_text_chunks(pdf_path_or_bytes):
    doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf") if isinstance(pdf_path_or_bytes, bytes) else fitz.open(pdf_path_or_bytes)
    chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if not text.strip(): continue
        
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "page": page_num + 1})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def process_pdf(pdf_bytes):
    with st.spinner("Processing PDF..."):
        chunks = extract_text_chunks(pdf_bytes)
        if not chunks:
            st.error("No text found in PDF.")
            return
        
        texts = [c["text"] for c in chunks]
        metadatas = [{"page": c["page"]} for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        embeddings = st.session_state.embedder.encode(texts).tolist()
        
        st.session_state.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        st.session_state.pdf_processed = True
        st.success("PDF processed successfully!")
        
        summary_context = "\n".join(texts[:5]) 
        st.session_state.pdf_summary = generate_summary(summary_context)

def generate_summary(context):
    token = get_hf_token()
    if not token:
        return "Could not generate summary because HF_TOKEN is missing."
    
    prompt = f"""You are a helpful study assistant. Write a short, 3-4 sentence summary of the following document so the user knows what it is about.
    
Context:
{context}
"""
    last_error = None
    for model in LLM_MODELS:
        client = InferenceClient(model, token=token)
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue
    return f"Failed to generate summary. Last error: {last_error}"

def query_llm(context, question):
    token = get_hf_token()
    if not token:
        st.error("HF_TOKEN not found. Please set it in secrets or environment to use the LLM.")
        return None
    
    prompt = f"""You are a helpful study assistant. Use the following context from the user's notes to answer the question.
If the context does not contain the answer, say exactly "I couldn't find that in your notes." and nothing else.
If you find the answer, include quotes or snippets from the context to support your answer and mention the source page numbers.

Context:
{context}

Question:
{question}
"""
    last_error = None
    for model in LLM_MODELS:
        client = InferenceClient(model, token=token)
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            if model != LLM_MODELS[0]:
                st.info(f"Model fallback activated. Using {model}")
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue
            
    if last_error:
        err_msg = str(last_error).lower()
        if "429" in err_msg or "rate limit" in err_msg:
            st.error("You've hit the Hugging Face API rate limit. Please wait a moment and try again.")
        else:
            st.error(f"Error calling HF API (all fallback models failed). Last error: {last_error}")
    return None

def generate_flashcards(context):
    token = get_hf_token()
    if not token:
        st.error("HF_TOKEN not found. Please set it in secrets or environment to use the LLM.")
        return []
    
    prompt = f"""Generate 5 study flashcards based on the following context. 
Output ONLY valid JSON in this exact format, with no markdown formatting or extra text:
[
  {{"q": "question here", "a": "answer here"}}
]

Context:
{context}
"""
    last_error = None
    for model in LLM_MODELS:
        client = InferenceClient(model, token=token)
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            content = response.choices[0].message.content
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                content = content[start_idx:end_idx]
                
            cards = json.loads(content.strip())
            
            if model != LLM_MODELS[0]:
                st.info(f"Model fallback activated. Using {model}")
            return cards
        except Exception as e:
            last_error = e
            continue
            
    if last_error:
        err_msg = str(last_error).lower()
        if "429" in err_msg or "rate limit" in err_msg:
            st.error("You've hit the Hugging Face API rate limit. Please wait a moment and try again.")
        else:
            st.error(f"Failed to generate flashcards (all fallback models failed). Last error: {last_error}")
    return []

def review_card(card, quality):
    if quality < 3:
        card['reps'] = 0
        card['interval'] = 1
    else:
        if card['reps'] == 0:
            card['interval'] = 1
        elif card['reps'] == 1:
            card['interval'] = 6
        else:
            card['interval'] = round(card['interval'] * card['efactor'])
        card['reps'] += 1
    
    card['efactor'] = max(1.3, card['efactor'] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    card['due'] = (datetime.datetime.now() + datetime.timedelta(days=card['interval'])).timestamp()
    return card

# -- UI --
st.sidebar.title("RecalLens")
page = st.sidebar.radio("Navigation", ["Upload & Ask", "Flashcards", "About"])

if page == "Upload & Ask":
    st.header("Ask Your Notes")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload PDF Notes", type=["pdf"])
    with col2:
        st.write("Or try a sample:")
        if st.button("Use sample notes"):
            try:
                with open("sample_notes.pdf", "rb") as f:
                    process_pdf(f.read())
            except Exception as e:
                st.error("Sample notes not found.")
                
    if uploaded_file is not None and not st.session_state.pdf_processed:
        if st.button("Process Uploaded PDF"):
            process_pdf(uploaded_file.read())
            st.rerun()
            
    if st.session_state.pdf_processed:
        if st.session_state.pdf_summary:
            st.info(f"**Document Summary:**\n{st.session_state.pdf_summary}")
            
        st.write("---")
        
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["q"])
            with st.chat_message("assistant"):
                st.write(chat["a"])
                if chat.get("sources"):
                    with st.expander("Sources Context"):
                        for doc, meta in zip(chat["sources"]["docs"], chat["sources"]["metas"]):
                            st.write(f"**Page {meta['page']}**: {doc}")
                            
        question = st.chat_input("Ask a question about your notes...")
        if question:
            with st.chat_message("user"):
                st.write(question)
                
            with st.spinner("Searching..."):
                q_emb = st.session_state.embedder.encode([question]).tolist()
                results = st.session_state.collection.query(
                    query_embeddings=q_emb,
                    n_results=4
                )
                
                if results['documents'] and len(results['documents'][0]) > 0:
                    context_str = ""
                    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                        context_str += f"[Page {meta['page']}]: {doc}\n\n"
                    
                    answer = query_llm(context_str, question)
                    if answer:
                        with st.chat_message("assistant"):
                            st.write(answer)
                            with st.expander("Sources Context"):
                                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                                    st.write(f"**Page {meta['page']}**: {doc}")
                                    
                        st.session_state.chat_history.append({
                            "q": question, 
                            "a": answer, 
                            "sources": {"docs": results['documents'][0], "metas": results['metadatas'][0]}
                        })
                else:
                    st.write("No relevant context found.")

elif page == "Flashcards":
    st.header("Flashcard Review")
    
    if st.session_state.pdf_processed:
        if st.button("Generate 5 New Flashcards"):
            with st.spinner("Generating flashcards..."):
                results = st.session_state.collection.get(limit=10)
                if results['documents']:
                    context = "\n".join(results['documents'])
                    new_cards = generate_flashcards(context)
                    for c in new_cards:
                        st.session_state.flashcards.append({
                            "q": c.get("q", "Unknown"),
                            "a": c.get("a", "Unknown"),
                            "efactor": 2.5,
                            "interval": 0,
                            "reps": 0,
                            "due": datetime.datetime.now().timestamp()
                        })
                    st.success(f"Generated {len(new_cards)} flashcards!")
    else:
        st.info("Please process a PDF in 'Upload & Ask' first to generate flashcards.")
        
    if st.session_state.flashcards:
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Question', 'Answer'])
        for c in st.session_state.flashcards:
            writer.writerow([c['q'], c['a']])
            
        st.download_button(
            label="📥 Download Flashcards (CSV)",
            data=output.getvalue(),
            file_name="recallens_flashcards.csv",
            mime="text/csv"
        )
        
    now = datetime.datetime.now().timestamp()
    due_cards = [c for c in st.session_state.flashcards if c['due'] <= now]
    
    st.write(f"**Total cards:** {len(st.session_state.flashcards)} | **Due for review:** {len(due_cards)}")
    
    if due_cards:
        card = due_cards[0]
        st.write("---")
        st.write("### Question")
        st.write(card["q"])
        
        if "show_answer" not in st.session_state:
            st.session_state.show_answer = False
            
        if not st.session_state.show_answer:
            if st.button("Show Answer"):
                st.session_state.show_answer = True
                st.rerun()
        else:
            st.write("### Answer")
            st.write(card["a"])
            st.write("How was it?")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Again (Hard)"):
                    review_card(card, 0)
                    st.session_state.show_answer = False
                    st.rerun()
            with c2:
                if st.button("Good"):
                    review_card(card, 3)
                    st.session_state.show_answer = False
                    st.rerun()
            with c3:
                if st.button("Easy"):
                    review_card(card, 5)
                    st.session_state.show_answer = False
                    st.rerun()
    else:
        if st.session_state.flashcards:
            st.success("You are all caught up!")

elif page == "About":
    st.header("About RecalLens")
    st.write("""
    RecalLens is a study companion for university students. 
    It leverages open-source technology to extract text, run semantic search, and provide answers and flashcards using spaced repetition.
    
    **Open-Source Stack:**
    - **LLM:** Llama 3.1 8B Instruct (with automatic fallback to Zephyr-7B) via Hugging Face
    - **Vector DB:** ChromaDB
    - **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
    - **UI:** Streamlit
    
    **Coming in Production:**
    - Neo4j Knowledge Graph integration
    - OCR for photographed notes (Tesseract)
    - Evaluation harness
    """)
