import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = "uploads"
VECTOR_DB_PATH = "faiss_index"
MAX_FILE_SIZE_MB = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

# -----------------------------
# LLM + EMBEDDINGS
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3
)

embeddings = OpenAIEmbeddings()

# -----------------------------
# RUNTIME OBJECTS
# -----------------------------
vectorstore = None
qa_chain = None
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -----------------------------
# PDF PROCESSING
# -----------------------------
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    store = FAISS.from_documents(split_docs, embeddings)
    store.save_local(VECTOR_DB_PATH)

    return store

def load_vectorstore():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings)
    return None

def init_chain():
    global qa_chain
    if not vectorstore:
        raise ValueError("Upload a PDF first")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        chain_type="stuff"
    )

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "routes": ["/upload", "/chat", "/reset"]
    })

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global vectorstore

    if "file" not in request.files:
        return jsonify({"error": "File missing"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDFs allowed"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    vectorstore = process_pdf(path)
    init_chain()

    return jsonify({"message": "PDF processed successfully"})

@app.route("/chat", methods=["POST"])
def chat():
    if not qa_chain:
        return jsonify({"error": "Upload PDF first"}), 400

    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query required"}), 400

    result = qa_chain.invoke({"question": query})
    return jsonify({"answer": result["answer"]})

@app.route("/reset", methods=["POST"])
def reset():
    global memory
    memory.clear()
    return jsonify({"message": "Conversation reset"})

# -----------------------------
# SERVER
# -----------------------------
if __name__ == "__main__":
    vectorstore = load_vectorstore()
    if vectorstore:
        init_chain()

    app.run(host="0.0.0.0", port=5050)
