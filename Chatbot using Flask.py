# Install dependencies first (if not installed)
# pip install openai langchain langchain-community faiss-cpu flask PyPDF2 tiktoken

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# 1️⃣ Setup
# -----------------------------

OPENAPI_API_KEY="add key here"

os.environ["OPENAI_API_KEY"] = OPENAPI_API_KEY
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# 2️⃣ Global Objects
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
vectorstore = None  # Will hold the FAISS store dynamically
qa_chain = None


# -----------------------------
# 3️⃣ Helper Function — Create Vectorstore from PDF
# -----------------------------
def process_pdf(file_path):
    """Extract text, split into chunks, and create FAISS vectorstore"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    store = FAISS.from_documents(docs, embeddings)
    return store


# -----------------------------
# 4️⃣ Initialize Chat Chain
# -----------------------------
def init_chain():
    global qa_chain
    if not vectorstore:
        raise ValueError("No document data loaded. Please upload a PDF first.")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        chain_type="stuff",
    )


# -----------------------------
# 5️⃣ API Endpoints
# -----------------------------

@app.route('/')
def home():
    return jsonify({
        "message": "LangChain Chatbot API is running.",
        "endpoints": ["/upload", "/chat", "/reset"]
    })


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process a PDF file"""
    global vectorstore

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Process PDF -> Vectorstore
    vectorstore = process_pdf(file_path)
    init_chain()

    return jsonify({"message": f"PDF '{filename}' uploaded and processed successfully."})


@app.route('/chat', methods=['POST'])
def chat():
    """Ask questions based on uploaded PDF"""
    global qa_chain
    if not qa_chain:
        return jsonify({"error": "Please upload a PDF first."}), 400

    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is missing."}), 400

    try:
        result = qa_chain({"question": query})
        answer = result["answer"]
        return jsonify({"query": query, "response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_memory():
    """Reset chatbot conversation memory"""
    global memory, qa_chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if qa_chain:
        qa_chain.memory = memory
    return jsonify({"message": "Chat memory has been reset."})


# -----------------------------
# 6️⃣ Run Flask Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
