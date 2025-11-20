from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import logging
from dotenv import load_dotenv
# Enable CORS (for frontend requests)
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded.")

# Initialize LLM with a faster model (gpt-oss-20b for quick, accurate responses)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",  # Ultra-fast & accurate for tutoring
    temperature=0.1  # Keeps it natural without rambling
)
logger.info("Groq LLM initialized with faster model.")

# Initialize memory (reduced k to 5 for shorter prompts and faster processing)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,  # Reduced from 7 to keep prompts shorter
    input_key="input",
    output_key="answer"
)

# Updated prompt for beginner English tutor focused on hobbies (simple English, shorter responses)
tutor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a friendly English tutor for beginners chatting about hobbies. dont talk about yourself talk about more about him.
     Keep it fun and engaging—share a quick, simple to connect. 
     Always use very basic, easy English in all responses. 
     If the user says something incorrect (grammar, word choice, or simple facts), gently correct by starting with 'You mean [corrected version]?' then explain briefly and positively. 
     For example: 'You mean "I like painting pictures"? Great choice—painting helps relax!' 
     Otherwise, gently fix errors (e.g., 'Hey, it's "a little bit" – no worries!') and flow on naturally. 
     Suggest smoother phrases sometimes, like 'Try saying... for a chill vibe.' 
     Build on past chats from history. 
     Respond like a relaxed friend: warm, natural, super short—1-3 sentences only."""), 
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # You can replace "*" with your frontend URL (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allow all headers
)

# Pydantic model for ask endpoint (removed audio flag)
class AskRequest(BaseModel):
    question: str

# Health check endpoint
@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    # return HTMLResponse(open("index.html").read())
    return JSONResponse({
            "message": "hello"
        })

# Ask endpoint - Using tutor prompt only (text-only, no TTS for speed)
@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        # Load chat history from memory
        chat_vars = memory.load_memory_variables({})
        chat_history = chat_vars.get("chat_history", [])
       
        # Format tutor prompt with history and input
        messages = tutor_prompt.format_messages(
            chat_history=chat_history, 
            input=request.question
        )
       
        # Invoke LLM (this is the main potential bottleneck, but faster model helps)
        result = llm.invoke(messages)
        answer = result.content
       
        logger.info("Q: %s", request.question)
        logger.info("A: %s", answer)
       
        # Save to memory
        memory.save_context({"input": request.question}, {"answer": answer})
       
        return JSONResponse({
            "question": request.question,
            "answer": answer,
        })
    except Exception as e:
        logger.error("Error during QA: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
