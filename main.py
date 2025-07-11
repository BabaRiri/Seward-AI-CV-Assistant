import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

#Load and process CV
script_dir = os.path.dirname(os.path.abspath(__file__))
cv_path = os.path.join(script_dir, 'cv.txt')

if os.path.exists(cv_path):
    try:
        loader = TextLoader(cv_path, encoding='utf-8')
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        print("CV loaded successfully.")
    except Exception as e:
        print(f"Error loading CV file: {str(e)}")
        documents = []
else:
    print(f"CV file not found at: {cv_path}")
    documents = []