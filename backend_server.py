import os
import uuid
import shutil
import json
import asyncio
import threading
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
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

# ==============================================================================
# WebSocket 任务管理 (用于实时进度推送)
# ==============================================================================
# 存储所有活跃的生成任务: task_id -> {progress, stage, message, complete, video_url, error}
active_tasks = {}
active_tasks_lock = threading.Lock()


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


# ==============================================================================
# WebSocket 进度推送 (替代 SSE)
# ==============================================================================
@app.post("/generate-ws")
async def generate_video_ws(
    text: str = Form(...),
    image: UploadFile = File(...)
):
    """
    上传图片并创建生成任务，返回 task_id。
    前端随后通过 WebSocket /ws/{task_id} 连接以接收实时进度。
    """
    # 1. 保存上传的图片
    file_extension = os.path.splitext(image.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    image_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    output_filename = f"gen_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    task_id = str(uuid.uuid4())

    # 2. 初始化任务状态
    with active_tasks_lock:
        active_tasks[task_id] = {
            "progress": 0.0,
            "stage": "start",
            "message": "任务已创建，准备生成...",
            "complete": False,
            "video_url": None,
            "error": None,
            "text": text,
            "image_filename": unique_filename,
            "video_filename": output_filename,
        }

    # 3. 在后台线程中启动生成
    def run_generation():
        def progress_callback(percent, stage, message):
            """GPU流水线中的同步回调 → 更新 active_tasks"""
            with active_tasks_lock:
                if task_id in active_tasks:
                    active_tasks[task_id]["progress"] = percent
                    active_tasks[task_id]["stage"] = stage
                    active_tasks[task_id]["message"] = message
            # 调试打印: 后端发送进度信号
            print(f"[DEBUG][WS Backend] 发送进度: task={task_id[:8]}, percent={percent:.1f}%, stage={stage}, msg={message}")

        try:
            print(f"[*] Backend: WebSocket generation for text: '{text}', task_id={task_id[:8]}...")

            result_path = generate_video_from_text(
                text, image_path, output_path, progress_callback=progress_callback
            )

            if result_path and os.path.exists(result_path):
                # 保存项目记录
                project_data = {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "image_filename": unique_filename,
                    "video_filename": output_filename,
                    "video_url": f"/outputs/{output_filename}",
                    "created_at": datetime.now().isoformat(),
                }
                save_project(project_data)

                with active_tasks_lock:
                    if task_id in active_tasks:
                        active_tasks[task_id]["progress"] = 100
                        active_tasks[task_id]["stage"] = "complete"
                        active_tasks[task_id]["message"] = "视频生成成功!"
                        active_tasks[task_id]["complete"] = True
                        active_tasks[task_id]["video_url"] = f"/outputs/{output_filename}"
                print(f"[DEBUG][WS Backend] 任务完成: task={task_id[:8]}, video_url=/outputs/{output_filename}")
            else:
                with active_tasks_lock:
                    if task_id in active_tasks:
                        active_tasks[task_id]["error"] = "Video generation failed"
                        active_tasks[task_id]["complete"] = True
                print(f"[DEBUG][WS Backend] 任务失败: task={task_id[:8]} (no output)")
        except Exception as e:
            print(f"[-] Error during WebSocket generation: {e}")
            with active_tasks_lock:
                if task_id in active_tasks:
                    active_tasks[task_id]["error"] = str(e)
                    active_tasks[task_id]["complete"] = True
            print(f"[DEBUG][WS Backend] 任务异常: task={task_id[:8]}, error={e}")

    import threading as th
    thread = th.Thread(target=run_generation, daemon=True)
    thread.start()

    return {
        "success": True,
        "task_id": task_id,
        "message": "任务已创建，请通过 WebSocket 连接获取进度"
    }


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket 端点: 前端连接后，持续推送生成进度直到任务完成
    """
    await websocket.accept()
    print(f"[DEBUG][WS Backend] WebSocket 客户端已连接: task={task_id[:8]}")

    try:
        # 发送初始状态
        with active_tasks_lock:
            if task_id in active_tasks:
                task = active_tasks[task_id]
                await websocket.send_json({
                    "type": "progress",
                    "percent": task["progress"],
                    "stage": task["stage"],
                    "message": task["message"],
                })
                print(f"[DEBUG][WS Backend] 发送初始状态: task={task_id[:8]}, percent={task['progress']:.1f}%, stage={task['stage']}")
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"未找到任务: {task_id}"
                })
                print(f"[DEBUG][WS Backend] 未找到任务: task={task_id[:8]}")
                await websocket.close()
                return

        # 持续轮询任务状态并推送 (100ms 间隔)
        last_percent = -1
        last_stage = ""
        while True:
            await asyncio.sleep(0.1)  # 100ms 轮询间隔

            with active_tasks_lock:
                if task_id not in active_tasks:
                    break
                task = active_tasks[task_id]
                current_percent = task["progress"]
                current_stage = task["stage"]
                current_message = task["message"]
                is_complete = task["complete"]
                error = task["error"]
                video_url = task["video_url"]

            # 仅在进度或阶段变化时发送，避免重复推送
            if current_percent != last_percent or current_stage != last_stage:
                await websocket.send_json({
                    "type": "progress",
                    "percent": current_percent,
                    "stage": current_stage,
                    "message": current_message,
                })
                print(f"[DEBUG][WS Backend] 推送进度: task={task_id[:8]}, percent={current_percent:.1f}%, stage={current_stage}, msg={current_message}")
                last_percent = current_percent
                last_stage = current_stage

            if is_complete:
                if error:
                    await websocket.send_json({
                        "type": "error",
                        "message": error,
                    })
                    print(f"[DEBUG][WS Backend] 推送错误: task={task_id[:8]}, error={error}")
                elif video_url:
                    await websocket.send_json({
                        "type": "complete",
                        "video_url": video_url,
                        "message": "视频生成成功!",
                    })
                    print(f"[DEBUG][WS Backend] 推送完成: task={task_id[:8]}, video_url={video_url}")
                break

        await websocket.close()
        print(f"[DEBUG][WS Backend] WebSocket 连接已关闭: task={task_id[:8]}")

    except WebSocketDisconnect:
        print(f"[DEBUG][WS Backend] 客户端断开连接: task={task_id[:8]}")
    except Exception as e:
        print(f"[DEBUG][WS Backend] WebSocket 错误: task={task_id[:8]}, error={e}")
        try:
            await websocket.close()
        except Exception:
            pass

    # 清理完成的任务 (保留一段时间)
    with active_tasks_lock:
        if task_id in active_tasks and active_tasks[task_id]["complete"]:
            del active_tasks[task_id]
            print(f"[DEBUG][WS Backend] 已清理完成任务: task={task_id[:8]}")


@app.get("/")
async def root():
    return {"message": "Lip Sync Backend is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
