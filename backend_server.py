import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from lip_sync.inference_pipeline import generate_video_from_text

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "output/text_driven_result"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files to serve the generated videos and uploaded images
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

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
        
        # This calls the core logic we researched earlier
        result_path = generate_video_from_text(text, image_path, output_path)
        
        if result_path and os.path.exists(result_path):
            # Return the URL to the generated video
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
