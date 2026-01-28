from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.youtube_rag import build_rag_chain
from urllib.parse import urlparse, parse_qs

app = FastAPI()

rag_chain = None


# --------- Helper function ----------
def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)

    # Case 1: https://www.youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get("v")
        if video_id:
            return video_id[0]

    # Case 2: https://youtu.be/VIDEO_ID
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")

    raise HTTPException(status_code=400, detail="Invalid YouTube URL")


# --------- Request Models ----------
class YouTubeRequest(BaseModel):
    youtube_url: str


class QuestionRequest(BaseModel):
    question: str


# --------- Endpoints ----------
@app.post("/youtube/ingest")
def ingest(req: YouTubeRequest):
    global rag_chain

    video_id = extract_video_id(req.youtube_url)
    rag_chain = build_rag_chain(video_id)

    return {
        "status": "video processed",
        "video_id": video_id
    }


@app.post("/youtube/ask")
def ask(req: QuestionRequest):
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="Video not processed yet")

    answer = rag_chain.invoke(req.question)
    return {"answer": answer}
