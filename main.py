from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.youtube_rag import build_rag_chain
import os

app = FastAPI(title="YouTube RAG API")

# Global chain storage
rag_chain = None
current_video_id = None

class YouTubeRequest(BaseModel):
    youtube_url: str

class QuestionRequest(BaseModel):
    question: str

@app.post("/youtube/ingest")
def ingest(req: YouTubeRequest):
    global rag_chain, current_video_id
    
    try:
        rag_chain = build_rag_chain(req.youtube_url)
        current_video_id = req.youtube_url
        return {"status": "video processed", "video_id": current_video_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing failed: {str(e)}")

@app.post("/youtube/ask")
def ask(req: QuestionRequest):
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="Process video first using /ingest")
    
    try:
        answer = rag_chain.invoke(req.question)
        return {"answer": answer, "video_id": current_video_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "video_loaded": rag_chain is not None}
