from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from .config.redis_config import queue
from .core.agent_worker import ingest_resume, generate_hiring_report
import shutil
import os

app = FastAPI(title="Lumina - AI Talent Scout")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend
app.mount("/static", StaticFiles(directory="lumina/static"), name="static")

@app.get("/")
def root():
    return {"message": "Lumina API is running. Visit /static/index.html"}

@app.post("/upload")
async def upload_files(
    job_description: str = Form(...),
    resumes: List[UploadFile] = File(...)
):
    """
    1. Ingests resumes (Embed & Store).
    2. Enqueues a job to generate the Hiring Report.
    """
    
    # 1. Process Resumes (Synchronously for simplicity in this demo, or queue each)
    # Ideally, we queue each ingestion. Here we do it inline to ensure data is ready 
    # before the report generation starts, or we chain jobs. 
    # For robust production, use a flow/dag.
    
    saved_count = 0
    for resume in resumes:
        content = await resume.read()
        # In a real app, queue this. Here we call it directly to populate DB first.
        success = ingest_resume(content, resume.filename) 
        if success:
            saved_count += 1
            
    # 2. Queue the Report Generation
    job = queue.enqueue(generate_hiring_report, job_description)
    
    return {
        "status": "Resumes Processed & Report Generation Queued",
        "processed_count": saved_count,
        "job_id": job.id
    }

@app.get("/results/{job_id}")
def get_results(job_id: str):
    job = queue.fetch_job(job_id)
    if not job:
        return {"status": "Job not found"}
    
    if job.is_finished:
        return {"status": "finished", "report": job.return_value()}
    elif job.is_failed:
        return {"status": "failed"}
    else:
        return {"status": "processing"}
