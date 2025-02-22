import os
from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from typing import Optional


groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_API_KEY')

class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.retrieval_chain =  None

    def create_index(self, path:str, index_save_path:str = 'vector_db_index') -> FAISS:
        
        """Create and save FAISS index from PDF"""

        try: 
            loader = PyPDFLoader(path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
            final_docs = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(
                model_name = 'sentence-transformers/all-MiniLM-L6-v2',
            )
            self.vectorstore = FAISS.from_documents(final_docs, embeddings)
            #save the vectorstore
            self.vectorstore.save_local(index_save_path)

            return self.vectorstore
        except Exception as e:
            raise RuntimeError(f"Index Creation Failed:{str(e)}")
        

    def load_index(self, index_path: str = 'vector_db_index') -> Optional[FAISS]:
        
        """Load existing FAISS index from disk"""
        try:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            self.vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            return self.vectorstore
        except Exception as e:
            print(f"Error: Index file not found at {index_path}")
            return None  # Or raise the exception, depending on how you want to handle it
        except Exception as e:
            raise RuntimeError(f"Index Loading Failed:{str(e)}")
    

    def initialize_retrieval_chain(self):

        """Initialize the RAG retrieval chain"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        llm = ChatGroq(
            model='llama3-8b-8192',
            api_key = groq_api_key
        )
        
        prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        self.retrieval_chain = create_retrieval_chain(
            self.vectorstore.as_retriever(), combine_docs_chain
        )
        return self.retrieval_chain

    def generate_response(self, query:str) -> str:
         """Generate response using RAG"""
         if not self.retrieval_chain:
             raise ValueError("Retrieval chain not initialized")
         response = self.retrieval_chain.invoke({'input': query})
         return response['answer']
         

rag_system = RAGSystem()