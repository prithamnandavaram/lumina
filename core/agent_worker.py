"""
Lumina Agent Worker
===================

This module serves as the "Brain" of the Lumina application.
It handles the background processing of resumes and the AI reasoning loop.

Core Responsibilities:
1. Ingest Resumes: Read PDFs, chunk text, and store Embeddings in Qdrant.
2. Hire Agent: A ReAct (Reason+Act) Agent that uses Groq (Llama-3) to:
   - Analyze Job Descriptions.
   - Search the Qdrant vector database for matching candidates.
   - Generate a structured Hiring Report.

Tech Stack:
- LLM: Groq (Llama-3-70b)
- Vector DB: Qdrant
- Embeddings: HuggingFace (Local)
"""
import os
import io
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Any
from toon import encode as toon_encode

load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6334")
COLLECTION_NAME = "resumes"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Clients ---
# 1. Embeddings (Local)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 2. Qdrant Client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Ensure collection exists
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# 3. Vector Store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# 4. LLM Client (Groq via OpenAI)
llm_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# --- Output Format for Agent ---
class AgentStep(BaseModel):
    step: str = Field(..., description="The ID of the step. Example: PLAN, TOOL, OUTPUT")
    content: Optional[Any] = Field(None, description="The content/reasoning for the step")
    tool: Optional[str] = Field(None, description="The tool to call")
    input: Optional[str] = Field(None, description="The input for the tool")

# --- Tools ---
def search_resumes(query: str):
    """Searches the vector DB for resumes matching the query. Returns TOON format."""
    print(f"🔎 Searching resumes for: {query}")
    results = vector_store.similarity_search(query, k=3)
    
    # Prepare data for official TOON serialization
    data = []
    for res in results:
        data.append({
            "source": res.metadata.get("source", "Unknown"),
            "snippet": res.page_content[:400]
        })
        
    return toon_encode({"resumes": data})

available_tools = {
    "search_resumes": search_resumes
}

# --- Core Logic ---

def ingest_resume(file_content: bytes, filename: str):
    """Extracts text from PDF and stores embedding."""
    try:
        print(f"📄 Processing: {filename}")
        reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
        vector_store.add_texts(chunks, metadatas=metadatas)

        print(f"✅ Stored {len(chunks)} embeddings for: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return False

def generate_hiring_report(job_description: str):
    """Agentic workflow to match JD to Resumes."""
    print("🤖 Starting Hiring Agent...")
    
    SYSTEM_PROMPT = """
    You are 'Lumina', an expert AI Talent Scout.
    Your goal is to screen candidates for a specific Job Description (JD).
    
    You have access to a tool `search_resumes(query)` which searches the database of uploaded resumes.
    
    **PROTOCOL: TOON (Token-Oriented Object Notation) — Official Spec v3.0**
    The tool returns data in TOON format (Compact Tabular Array).
    TOON reduces token usage by 30-60% vs JSON by eliminating braces and quotes.
    Example:
    resumes[2,]{source,snippet}:
      resume1.pdf,Top skills: Python...
      resume2.pdf,Experience: 5 years...
    
    You must output your response in valid JSON format matching this schema:
    {
        "step": "PLAN" | "TOOL" | "OUTPUT",
        "content": "Your reasoning or the final report",
        "tool": "search_resumes" (only if step is TOOL),
        "input": "search query" (only if step is TOOL)
    }

    Protocol:
    1. PLAN: Analyze the JD.
    2. TOOL: Search for skills.
    3. OUTPUT: Final Report based on the TOON data received.
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the Job Description:\n{job_description}"}
    ]
    
    final_output = ""
    
    # ReAct Loop (Max 5 steps to prevent infinite loops)
    for _ in range(5):
        try:
            response = llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                # Groq doesn't support 'response_format' structured output well with OpenAI client compatibility in all versions yet, 
                # but let's try strict JSON mode or just parse textual output.
                # Ideally we ask for JSON in prompt, but here we used 'parse' before.
                # We will revert to simple generation and manual parsing for reliability with Groq/Llama or attempt json mode.
                # For simplicity here, sticking to JSON mode if possible or just standard prompting.
                # Let's switch to standard generation and manual parsing since structure output varies by provider.
                # HOWEVER, to keep it simple, we'll try json mode.
                response_format={"type": "json_object"}
            )
            
            # Since we removed 'parse', we load the json manually
            content_str = response.choices[0].message.content
            print(f"🔹 Model Raw Output: {content_str}") # Debug print
            step_data = AgentStep.model_validate_json(content_str)
            
            # Log the thought process
            if step_data.step == "PLAN":
                print(f"🧠 PLAN: {step_data.content}")
                messages.append({"role": "assistant", "content": content_str}) # Append raw JSON
            
            elif step_data.step == "TOOL":
                tool_name = step_data.tool
                tool_arg = step_data.input
                print(f"🛠️ TOOL: {tool_name}('{tool_arg}')")
                messages.append({"role": "assistant", "content": content_str}) # Append raw JSON
                
                if tool_name in available_tools:
                    tool_result = available_tools[tool_name](tool_arg)
                    messages.append({"role": "user", "content": f"TOOL_OUTPUT: {tool_result}"})
                else:
                    messages.append({"role": "user", "content": "Error: Tool not found."})
            
            elif step_data.step == "OUTPUT":
                print("📝 OUTPUT generated.")
                if isinstance(step_data.content, (dict, list)):
                    final_output = json.dumps(step_data.content, indent=2)
                else:
                    final_output = str(step_data.content)
                break
                
        except Exception as e:
            print(f"⚠️ Agent Error: {e}")
            break
            
    return final_output
