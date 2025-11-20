from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,
    input_key="input",
    output_key="answer",
)

# Prompt
tutor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a friendly English tutor for beginners chatting about hobbies.
     Use very simple English, very short answers (1–3 sentences).
     Do not talk about yourself; talk more about the user.
     If the user makes a mistake, correct gently:
     Start with: 'You mean "...?"' and give the corrected version.
     Keep conversation warm and friendly."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Lazy-load LLM for Vercel (prevents cold-start crashes)
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise Exception("Missing GROQ_API_KEY in environment variables")

    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",  # fast + stable for Vercel
        temperature=0.1
    )


# Models
class AskRequest(BaseModel):
    question: str


# ROUTES
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve index.html from /public"""
    index_path = os.path.join(os.path.dirname(__file__), "../public/index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    return FileResponse(index_path)


@app.post("/ask")
async def ask_question(req: AskRequest):
    try:
        # Get chat history
        chat_vars = memory.load_memory_variables({})
        chat_history = chat_vars.get("chat_history", [])

        # Build prompt
        messages = tutor_prompt.format_messages(
            chat_history=chat_history,
            input=req.question
        )

        # Lazy-load LLM each request
        llm = get_llm()

        # Generate response
        result = llm.invoke(messages)
        answer = result.content

        # Save memory
        memory.save_context(
            {"input": req.question},
            {"answer": answer}
        )

        return JSONResponse({
            "question": req.question,
            "answer": answer
        })

    except Exception as e:
        logger.error(f"Server error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Debug endpoint (optional)
@app.get("/debug")
async def debug():
    return {
        "cwd": os.getcwd(),
        "files": os.listdir(os.getcwd()),
        "GROQ_API_KEY_loaded": bool(os.getenv("GROQ_API_KEY")),
    }
