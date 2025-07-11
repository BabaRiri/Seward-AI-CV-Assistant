import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def load_chatbot(cv_path: str):
    if not os.path.exists(cv_path):
        raise FileNotFoundError("CV file not found at: " + cv_path)

    # Load and chunk CV
    loader = TextLoader(cv_path, encoding="utf-8")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embeddings + vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # LLM + Memory
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Prompt template
    template = """
    You are Seward's AI assistant, helping people learn about Seward Mupereri's professional background. 
    You should be able to answer typical interview questions about Seward's professional, educational and personal background
    Your responses should be:
        - Friendly and approachable, but professional
        - Concise (2-3 sentences max for most responses)
        - Use a conversational tone, as if you're having with a CEO who you are trying to convince and show that Seward is skilled and qualified for a role.
        - If you don't know the answer, be honest but helpful
        - Feel free to show enthusiasm when talking about Seward's achievements
        
    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # Return the chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
