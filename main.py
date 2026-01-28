from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.youtube_rag import build_rag_chain

app = FastAPI()

rag_chain = None


class YouTubeRequest(BaseModel):
    youtube_url: str


class QuestionRequest(BaseModel):
    question: str


@app.post("/youtube/ingest")
def ingest(req: YouTubeRequest):
    global rag_chain
    try:
        rag_chain = build_rag_chain(req.youtube_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "video processed"}


@app.post("/youtube/ask")
def ask(req: QuestionRequest):
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="Video not processed yet")
    return {"answer": rag_chain.invoke(req.question)}
