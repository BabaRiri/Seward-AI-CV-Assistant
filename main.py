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

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        custom_prompt_template = """
            You are Seward Mupereri's CV AI Assistant, helping hiring managers learn about Seward Mupereri's professional background.
            Your reponses should be:
            - Professional but friendly and approachable
            - Concise and easy to understand
            - If you dont know the answer be honest but helpful

            Context:
            {context}

            Question:
            {question}

            Helpful Answer:
            """
        prompt = PromptTemplate(input_variables=["context", "question"], template=custom_prompt_template)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={
                'prompt': prompt
            },
            return_source_documents=True,
            output_key="answer"
        )
        print("CV loaded successfully.")
        chat_history = []
        print("Ask me anything about Seward's CV")

        while True:
            user_input = input("\nYour question: ")

            if user_input.lower() == "exit":
                print("bye!")
                
            if user_input.strip():
                try:
                    response = chain.invoke({"question": user_input, "chat_history": chat_history})
                    print("\nAssistant:", response["answer"])
                    # print("\nSource Documents:", response["source_documents"])
                except Exception as e:
                    print(f"Error processing your question: {str(e)}")
                
    except Exception as e:
        print(f"Error loading CV file: {str(e)}")
        documents = []
else:
    print(f"CV file not found at: {cv_path}")
    documents = []