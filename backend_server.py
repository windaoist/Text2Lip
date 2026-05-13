import os
import uuid
import shutil
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from lip_sync.inference_pipeline import generate_video_from_text

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "output/text_driven_result"
PROJECTS_DB = "data/projects.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files to serve the generated videos
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def load_projects():
    """Load projects from JSON database"""
    if os.path.exists(PROJECTS_DB):
        with open(PROJECTS_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_project(project_data):
    """Save a project record to the JSON database"""
    projects = load_projects()
    projects.append(project_data)
    with open(PROJECTS_DB, "w", encoding="utf-8") as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)


@app.get("/projects")
async def list_projects():
    """List all previously generated projects"""
    projects = load_projects()
    projects.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return {"success": True, "projects": projects}


@app.post("/generate")
async def generate_video(
    text: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # 1. Save the uploaded image
        file_extension = os.path.splitext(image.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        image_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 2. Run the lip-sync pipeline
        output_filename = f"gen_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"[*] Backend: Generating video for text: '{text}' with image: {image_path}")

        result_path = generate_video_from_text(text, image_path, output_path)

        if result_path and os.path.exists(result_path):
            # Save project record
            project_data = {
                "id": str(uuid.uuid4()),
                "text": text,
                "image_filename": unique_filename,
                "video_filename": output_filename,
                "video_url": f"/outputs/{output_filename}",
                "created_at": datetime.now().isoformat(),
            }
            save_project(project_data)

            return {
                "success": True,
                "video_url": f"/outputs/{output_filename}",
                "message": "Video generated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Video generation failed")

    except Exception as e:
        print(f"[-] Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Lip Sync Backend is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
