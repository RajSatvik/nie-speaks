import streamlit as st
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from supabase import create_client, Client

# -------------------- Streamlit Page Config --------------------
st.set_page_config(
    page_title="NIE Speaks",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': "https://i.pinimg.com/1200x/cc/14/d7/cc14d7c6d4576a0dda7d3e6b2ce6ed54.jpg",
        'Report a bug': "https://i.pinimg.com/1200x/cc/14/d7/cc14d7c6d4576a0dda7d3e6b2ce6ed54.jpg",
        'About': "### NIE Speaks - Your AI Assistant for NIE Information"
    }
)

# -------------------- Custom CSS --------------------
st.markdown("""
<meta name="google-site-verification" content="bYk4EloI6Pjl1kcqLbJoTTGsLaCzQ0FEfxddigbdeg8" />
<style>
    .main-header { font-size: 3rem; color: #2E4057; text-align: center; margin-bottom: 2rem; font-weight: bold; }
    section[data-testid="stSidebar"] { background-color: #2E4057 !important; color: #F5F7FA !important; }
    section[data-testid="stSidebar"] * { color: #F5F7FA !important; }
    .sidebar-title { font-size: 1.8rem; color: #FFFFFF; font-weight: bold; text-align: center; margin-bottom: 1rem; }
    .stButton button { background-color: #2E4057; color: white; border-radius: 10px; padding: 0.5rem 1rem; border: none; width: 100%; font-weight: bold; transition: all 0.3s ease; }
    .stButton button:hover { background-color: #1A2B3C; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(46,64,87,0.3); }
    .answer-box { background-color: #E8ECF1; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2E4057; margin-top: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #2E4057; line-height: 1.6; }
    .info-box { background-color: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #F5F7FA; margin: 0.5rem 0; color: #F5F7FA; }
    .developer-info { background-color: rgba(255,255,255,0.1); color: #F5F7FA; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 2rem; }
    .section-divider { border-top: 2px solid #F5F7FA; margin: 1.5rem 0; }
    .developer-info a { color: #F5F7FA; text-decoration: none; }
    .developer-info a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# -------------------- Supabase Setup --------------------
supabase: Client = create_client(
    st.secrets["supabase"]["url"],
    st.secrets["supabase"]["key"]
)

# -------------------- Core Logic --------------------
DATA_CONTENT = st.secrets["college_data"]

@st.cache_resource
def prepare_faiss():
    content = DATA_CONTENT
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=600,
        length_function=len,
        separators=['\n', '\n\n', ' ', ',']
    )
    chunks = splitter.split_text(content)
    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = FAISS.from_texts(chunks, embedding=embed_model)
    return faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retriever = prepare_faiss()

def answer_question_with_rag(user_query, faiss_retriever):
    cohere_model = ChatCohere(
        model="command-a-03-2025",
        temperature=0.9,
        cohere_api_key=st.secrets["cohere_api_key"]
    )
    
    rag_prompt = """
    Your name is Yamika, an AI assistant for The National Institute of Engineering (NIE), Mysuru.

    You have access to a CONTEXT section that contains information about NIE Mysuru.
    Your primary goal is to assist users with NIE-related queries.
    However, you should also be capable of handling general questions if the user asks something outside the context.

    --- RESPONSE RULES ---
    1. **NIE-related questions:** Use only the details from the CONTEXT section.
    2. **General questions (not related to NIE):** 
       - Provide a short and accurate general answer.
       - End your response with this line:
         "Please try asking questions related to NIE Mysuru for more relevant assistance."
    3. **Unanswerable questions (neither in context nor general knowledge):**
       - Say:
         "I couldn't find this information in my knowledge base. Please check the official NIE website or contact the administration for accurate details."
    4. **Comparison-based questions (e.g., 'Which is better?', 'Who is best?', 'Should I choose X or Y?'):**
       - Do NOT take sides or declare a winner.
       - Instead, give factual, objective information about both or all entities being compared.
       - Conclude with:
         "It depends on individual preferences, goals, and priorities."
    5. Keep responses factual, clear, concise, and student-friendly.

    --- CONTEXT ---
    {context}
    --- END CONTEXT ---

    --- QUESTION ---
    {question}
    --- END QUESTION ---

    Provide the best possible answer:
    """
    prompt_obj = PromptTemplate.from_template(rag_prompt)

    def concatenate_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    rag_pipeline = (
        {"context": faiss_retriever | concatenate_docs, "question": RunnablePassthrough()}
        | prompt_obj
        | cohere_model
        | StrOutputParser()
    )
    return rag_pipeline.invoke(user_query)

# -------------------- Main UI --------------------
def main():
    with st.sidebar:
        st.markdown('<p class="sidebar-title">NIE Speaks 🤖</p>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # About Section
        st.markdown("### About")
        st.markdown("""<div class="info-box">
        Say hi to <strong>NIE Speaks</strong> – your all-in-one AI guide for <em>The National Institute of Engineering, Mysuru!</em><br><br>
        This bot isn’t like the usual ChatGPT – it’s built to help you with <strong>everything about NIE</strong>.<br>
        From <strong>courses, exams, and admissions</strong> to the everyday stuff like <strong>Wi-Fi passwords, hostel do’s & don’ts, fests, canteen timings, and more</strong> – it’s got you covered.<br><br>
        😉 For freshers, it’s like having a friendly insider who knows it all!
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Tips for Better Results")
        st.markdown("""<div class="info-box">
        • Ask simple and direct questions<br>
        • One question at a time works best<br>
        • No question is “too small” – go ahead and ask!<br>
        • This bot is trained mainly on data from North Campus.<br>
        • Keep it NIE-related<br>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 💡 Contribute Information")
        st.markdown("""<div class="info-box">
        Know something useful about NIE that’s missing here?<br>
        Share it below and help improve the assistant! 🏫
        </div>""", unsafe_allow_html=True)

        with st.form("user_submission_form"):
            name = st.text_input("Your name (optional)")
            submission_text = st.text_area("Enter information you'd like to add")
            submitted = st.form_submit_button("Submit")

        if submitted:
            if len(submission_text.strip()) < 15:
                st.error("Please provide more detailed information.")
            else:
                try:
                    supabase.table("pending_submissions").insert({
                        "name": name if name else "Anonymous",
                        "content": submission_text.strip(),
                        "timestamp": datetime.now().isoformat()
                    }).execute()
                    st.success("✅ Thank you! Your submission has been recorded and will be reviewed.")
                except Exception as e:
                    st.error(f"Failed to submit: {str(e)}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("""<div class="developer-info">
        <h4>Developed by</h4>
        <h3><a href="https://www.linkedin.com/in/7vikraj/" target="_blank">Satvik Raj</a></h3>
        </div>""", unsafe_allow_html=True)

    # Main Chat Area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">NIE Speaks 🤖</h1>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("### Ask Your Question")
        question_input = st.text_area(
            "Enter your question about NIE:",
            placeholder="e.g., What are the engineering courses offered at NIE?",
            height=60,
            label_visibility="collapsed"
        )

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            ask_btn = st.button("🚀 Ask NIE Speaks", use_container_width=True)

        if ask_btn:
            if question_input.strip():
                with st.spinner("Hawa kitni acchi chal rhi hai yaar...🍃"):
                    try:
                        answer_text = answer_question_with_rag(question_input, retriever)
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        st.markdown("### Answer")
                        st.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.info("Try asking a different question.")
            else:
                st.warning("⚠️ Please enter a question to get started!")

# -------------------- Run App --------------------
if __name__ == "__main__":
    main()



