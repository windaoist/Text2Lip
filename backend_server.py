import os
import uuid
import shutil
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
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
    """
    Original sync endpoint for video generation.
    Kept for backward compatibility.
    """
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

        print(
            f"[*] Backend: Generating video for text: '{text}' with image: {image_path}")

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
            raise HTTPException(
                status_code=500, detail="Video generation failed")

    except Exception as e:
        print(f"[-] Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-stream")
async def generate_video_stream(
    text: str = Form(...),
    image: UploadFile = File(...)
):
    """
    SSE streaming endpoint for video generation with real-time progress.
    Returns progress events as Server-Sent Events.
    """
    # 1. Save the uploaded image
    file_extension = os.path.splitext(image.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    image_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    output_filename = f"gen_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    async def event_generator():
        """Async generator that yields SSE events"""
        queue = asyncio.Queue()
        completed = [False]
        result_data = [None]
        error_data = [None]

        def progress_callback(percent, stage, message):
            """Synchronous callback called from the GPU pipeline"""
            # Use run_coroutine_threadsafe to put events from sync to async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        queue.put({
                            "type": "progress",
                            "percent": percent,
                            "stage": stage,
                            "message": message,
                            "timestamp": datetime.now().isoformat()
                        }),
                        loop
                    )
            except RuntimeError:
                pass

        def run_generation():
            """Run the generation in a thread"""
            try:
                print(f"[*] Backend: Streaming generation for text: '{text}'")

                # Initial progress
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            queue.put({
                                "type": "progress",
                                "percent": 0,
                                "stage": "start",
                                "message": "开始生成...",
                                "timestamp": datetime.now().isoformat()
                            }),
                            loop
                        )
                except RuntimeError:
                    pass

                result_path = generate_video_from_text(
                    text, image_path, output_path, progress_callback=progress_callback
                )

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

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                queue.put({
                                    "type": "complete",
                                    "video_url": f"/outputs/{output_filename}",
                                    "message": "视频生成成功!"
                                }),
                                loop
                            )
                    except RuntimeError:
                        pass
                else:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                queue.put({
                                    "type": "error",
                                    "message": "Video generation failed"
                                }),
                                loop
                            )
                    except RuntimeError:
                        pass
            except Exception as e:
                print(f"[-] Error during streaming generation: {e}")
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            queue.put({
                                "type": "error",
                                "message": str(e)
                            }),
                            loop
                        )
                except RuntimeError:
                    pass
            finally:
                completed[0] = True

        # Start generation in a background thread
        import threading
        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        # Yield events from the queue
        while not completed[0] or not queue.empty():
            try:
                # Wait for events with a timeout so we can check completed status
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # No event yet, just continue the loop
                continue

        # Drain any remaining events
        while not queue.empty():
            try:
                event = queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.QueueEmpty:
                break

        # Send a final done event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/")
async def root():
    return {"message": "Lip Sync Backend is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
