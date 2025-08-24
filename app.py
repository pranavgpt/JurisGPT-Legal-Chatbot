import streamlit as st
import os
import time
import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="JurisGPT - Legal Case Analyzer", layout="wide")
st.title("🏛️ JurisGPT - Legal Case Decision Engine")

# Add sidebar for mode selection
st.sidebar.title("Analysis Mode")
analysis_mode = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["General Legal Chat", "Case Analysis & Decision", "IPC Section Lookup"]
)

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ffd0d0;
    }
    div.stButton > button:active {
        background-color: #ff6262;
    }
    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    .decision-box {
        background-color: #f0f8ff;
        color: #000000 !important;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4169e1;
        margin: 10px 0;
    }
    .ipc-section {
        background-color: #fff5ee;
        color: #000000 !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff6347;
        margin: 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Enhanced prompt templates for different modes
general_chat_template = """
<s>[INST]You are JurisGPT, a specialized legal assistant for Indian law. Provide accurate, concise information based on the Indian Penal Code and legal precedents.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}

Provide a clear, professional response based on Indian legal principles.
ANSWER:
</s>[INST]
"""

case_analysis_template = """
<s>[INST]You are JurisGPT, an expert legal case analyzer specializing in Indian Penal Code. Your task is to:

1. ANALYZE the case facts presented
2. IDENTIFY applicable IPC sections
3. DETERMINE the severity and classification of offenses
4. PROVIDE a structured legal decision with reasoning
5. SUGGEST likely outcomes based on Indian legal precedents

LEGAL CONTEXT: {context}
CHAT HISTORY: {chat_history}
CASE DETAILS: {question}

Structure your response as follows:
**CASE SUMMARY:**
[Brief summary of the case]

**APPLICABLE IPC SECTIONS:**
[List relevant sections with explanations]

**LEGAL ANALYSIS:**
[Detailed analysis of facts against law]

**DECISION & REASONING:**
[Your legal decision with supporting reasoning]

**LIKELY OUTCOME:**
[Probable court decision, penalties, bail considerations]

**PRECEDENTS:**
[Any relevant case precedents if available]

Provide a comprehensive legal analysis with decision-making based on Indian Penal Code.
ANSWER:
</s>[INST]
"""

ipc_lookup_template = """
<s>[INST]You are JurisGPT, an expert on Indian Penal Code. Provide detailed information about IPC sections, their applications, punishments, and related precedents.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUERY: {question}

Provide comprehensive information about the requested IPC section(s) including:
- Section details
- Punishment provisions
- Key elements of the offense
- Common applications
- Related sections

ANSWER:
</s>[INST]
"""

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.1)

# Function to create appropriate QA chain based on mode
def get_qa_chain(mode):
    if mode == "Case Analysis & Decision":
        prompt = PromptTemplate(template=case_analysis_template, input_variables=['context', 'question', 'chat_history'])
    elif mode == "IPC Section Lookup":
        prompt = PromptTemplate(template=ipc_lookup_template, input_variables=['context', 'question', 'chat_history'])
    else:
        prompt = PromptTemplate(template=general_chat_template, input_variables=['context', 'question', 'chat_history'])
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

# Case analysis helper function
def analyze_case_structure(response):
    """Extract structured information from case analysis response"""
    sections = {
        "summary": "",
        "ipc_sections": "",
        "analysis": "",
        "decision": "",
        "outcome": "",
        "precedents": ""
    }
    
    current_section = None
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if "CASE SUMMARY:" in line.upper():
            current_section = "summary"
        elif "APPLICABLE IPC SECTIONS:" in line.upper():
            current_section = "ipc_sections"
        elif "LEGAL ANALYSIS:" in line.upper():
            current_section = "analysis"
        elif "DECISION & REASONING:" in line.upper():
            current_section = "decision"
        elif "LIKELY OUTCOME:" in line.upper():
            current_section = "outcome"
        elif "PRECEDENTS:" in line.upper():
            current_section = "precedents"
        elif current_section and line:
            sections[current_section] += line + "\n"
    
    return sections

# Display mode-specific instructions
if analysis_mode == "Case Analysis & Decision":
    st.info("📋 **Case Analysis Mode**: Provide case facts and I'll analyze them against IPC provisions and suggest legal decisions.")
    
    with st.expander("💡 How to get better case analysis"):
        st.markdown("""
        **For optimal case analysis, include:**
        - Clear statement of facts
        - Actions of accused person(s)
        - Any evidence or circumstances
        - Specific questions about charges or defenses
        
        **Example:** "A person threatened someone with a knife and demanded money. What IPC sections apply and what would be the likely punishment?"
        """)

elif analysis_mode == "IPC Section Lookup":
    st.info("📚 **IPC Lookup Mode**: Ask about specific IPC sections, their applications, or legal concepts.")

# Get the appropriate QA chain
qa = get_qa_chain(analysis_mode)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        if message.get("role") == "assistant" and analysis_mode == "Case Analysis & Decision":
            # Try to display structured case analysis
            try:
                sections = analyze_case_structure(message.get("content"))
                if sections["decision"]:  # If we have structured analysis
                    if sections["summary"]:
                        st.markdown("### 📝 Case Summary")
                        st.markdown(sections["summary"])
                    
                    if sections["ipc_sections"]:
                        st.markdown("### ⚖️ Applicable IPC Sections")
                        st.markdown(f'<div class="ipc-section">{sections["ipc_sections"]}</div>', unsafe_allow_html=True)
                    
                    if sections["analysis"]:
                        st.markdown("### 🔍 Legal Analysis")
                        st.markdown(sections["analysis"])
                    
                    if sections["decision"]:
                        st.markdown("### 🏛️ Decision & Reasoning")
                        st.markdown(f'<div class="decision-box">{sections["decision"]}</div>', unsafe_allow_html=True)
                    
                    if sections["outcome"]:
                        st.markdown("### 📊 Likely Outcome")
                        st.markdown(sections["outcome"])
                    
                    if sections["precedents"]:
                        st.markdown("### 📚 Relevant Precedents")
                        st.markdown(sections["precedents"])
                else:
                    st.write(message.get("content"))
            except:
                st.write(message.get("content"))
        else:
            st.write(message.get("content"))

# Input prompt
input_prompt = st.chat_input("Describe your case or ask your legal question...")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("🧠 Analyzing legal case...", expanded=True):
            result = qa.invoke(input=input_prompt)
            
            if analysis_mode == "Case Analysis & Decision":
                # Display structured case analysis
                sections = analyze_case_structure(result["answer"])
                
                if sections["summary"]:
                    st.markdown("### 📝 Case Summary")
                    st.markdown(sections["summary"])
                
                if sections["ipc_sections"]:
                    st.markdown("### ⚖️ Applicable IPC Sections")
                    st.markdown(f'<div class="ipc-section">{sections["ipc_sections"]}</div>', unsafe_allow_html=True)
                
                if sections["analysis"]:
                    st.markdown("### 🔍 Legal Analysis")
                    st.markdown(sections["analysis"])
                
                if sections["decision"]:
                    st.markdown("### 🏛️ Decision & Reasoning")
                    st.markdown(f'<div class="decision-box">{sections["decision"]}</div>', unsafe_allow_html=True)
                
                if sections["outcome"]:
                    st.markdown("### 📊 Likely Outcome")
                    st.markdown(sections["outcome"])
                
                if sections["precedents"]:
                    st.markdown("### 📚 Relevant Precedents")
                    st.markdown(sections["precedents"])
                
                # If no structured sections found, display the full response
                if not any(sections.values()):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in result["answer"]:
                        full_response += chunk
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + " ▌")
            else:
                # Regular streaming response for other modes
                message_placeholder = st.empty()
                full_response = ""
                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")

        # Add control buttons
        col1, col2 = st.columns([1, 4])
        with col1:
            st.button('🗑️ Reset Chat', on_click=reset_conversation)
    
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# Add footer with usage tips
st.markdown("---")
st.markdown("💡 **Tip**: Switch between modes using the sidebar for different types of legal analysis!")