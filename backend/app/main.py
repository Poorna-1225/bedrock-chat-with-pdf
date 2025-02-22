from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from rag_backend import rag_system 
from redis_client import redis_client
from datetime import datetime
import os
import uuid
import json
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    try:
        #startup: connect to redis
        redis_client.connect()
        print("Connected to redis successfully")
        yield
    except Exception as e:
        raise RuntimeError(f"Failed to start the application: {str(e)}")
    finally:
        #shutdown: Cloase Redis connection
        if redis_client.client:
            redis_client.client.close()
            print("Redis connection closed")


app = FastAPI(lifespan=lifespan)


# Pydantic model for the query request
class QueryRequest(BaseModel):
    query: str
    session_id: str  # adding session id to request

class ChatHistoryResponse(BaseModel):
    messages: list

# Endpoint to upload a file and create the vector store index
@app.post("/upload-and-process", status_code= status.HTTP_201_CREATED)
async def upload_and_process(file: UploadFile = File(...)):
    try:
        # Generate unique filename to prevent collisions
        file_path = f"temp_{uuid.uuid4()}_{file.filename}"

        #save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # processing the uploaded file
        rag_system.create_index(file_path)
        rag_system.initialize_retrieval_chain()

        return {"message": "File processed and index created successfully"}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail=str(e))
    finally:
        #cleanup temp file even if error occurs
        if os.path.exists(file_path):
            os.remove(file_path)

# Endpoint to handle user queries
@app.post("/query")
async def query(request: QueryRequest):
    try:
        # Load the index
        if not rag_system.vectorstore:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail = "PDF not processed. Upload a file first"
            )
        
        # Ensure retrieval chain is initialized
        if not rag_system.retrieval_chain:
            rag_system.initialize_retrieval_chain()

        # Generate the response
        response = rag_system.generate_response(request.query)

        #store interaction in Redis
        redis_conn = redis_client.get_client()
        chat_entry = {
            "query": request.query,
            "respone": response,
            "timestamp": datetime.now().isoformat(),
        }
        redis_conn.rpush(f'chat:{request.session_id}', json.dumps(chat_entry))
        
        return {"response": response}
 
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail=str(e))
    
# Endpoint to retrieve chat history
@app.get("/chat/history/{session_id}", response_model= ChatHistoryResponse)
async def get_chat_history(session_id:str):
    try:
        redis_conn = redis_client.get_client()
        history = redis_conn.lrange(f"chat:{session_id}",0,-1)
        return {"messages": [json.loads(msg) for msg in history]}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail=str(e))
    
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True )