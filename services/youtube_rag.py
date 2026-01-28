from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()
# 2️⃣ Read API key from environment
API_KEY = os.getenv("GOOGLE_API_KEY")

# 3️⃣ Safety check (VERY IMPORTANT)
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment")



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(video_id: str):

    # 1. Transcript
    fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    transcript_list = fetched_transcript.to_raw_data()
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    #print(transcript)

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=API_KEY
    )

    # 5. Prompt
    prompt = PromptTemplate(
        template="""
        Answer ONLY from the context.
        If the answer is not present, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    # 6. Chain
    chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    rag_chain = chain | prompt | llm | StrOutputParser()

    return rag_chain





