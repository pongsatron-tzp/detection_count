import cv2
import time
import face_recognition
import numpy as np
import os
import datetime
import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import StreamingResponse
from pydantic_settings import BaseSettings
from dataclasses import dataclass, field
from typing import List, Any, Tuple, Dict
import uvicorn

# --- Configuration ---
class Settings(BaseSettings):
    CAMERA_INDEX: int = 0
    SAVE_DIR: str = "detected_faces"
    KNOWN_ENCODINGS_FILE: str = "known_faces.pkl"
    DETECTION_THRESHOLD: float = 0.70
    PROCESS_EVERY_N_FRAME: int = 2
    RECENTLY_SEEN_TIMEOUT: int = 10
    
    class Config:
        env_file = ".env"

settings = Settings()

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- State Management ---
@dataclass
class AppState:
    known_face_encodings: List[np.ndarray] = field(default_factory=list)
    recently_seen_unknowns: Dict[tuple, float] = field(default_factory=dict)
    frame_with_boxes: Any = None
    
    @property
    def known_faces_count(self) -> int:
        return len(self.known_face_encodings)

app_state = AppState()

# --- Global Controls ---
_camera_capture = None
_detection_task = None
_stop_detection_flag = asyncio.Event()
_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)

# --- Helper Functions ---
def save_new_face(frame: np.ndarray, face_location: tuple):
    try:
        top, right, bottom, left = face_location
        padding = 30
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(frame.shape[0], bottom + padding)
        right = min(frame.shape[1], right + padding)
        
        face_image = frame[top:bottom, left:right]
        if face_image.size == 0:
            logger.warning("Cropped face image has zero size, skipping save.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(settings.SAVE_DIR, f"face_{timestamp}.jpg")
        cv2.imwrite(filename, face_image)
        logger.info(f"Saved new face to: {filename}")
    except Exception as e:
        logger.error(f"Error saving new face: {e}")

def _invalidate_encoding_cache():
    if os.path.exists(settings.KNOWN_ENCODINGS_FILE):
        try:
            os.remove(settings.KNOWN_ENCODINGS_FILE)
            logger.info(f"Invalidated encoding cache file: '{settings.KNOWN_ENCODINGS_FILE}'")
        except OSError as e:
            logger.error(f"Failed to remove cache file: {e}")

def load_known_faces_sync():
    if os.path.exists(settings.KNOWN_ENCODINGS_FILE):
        logger.info(f"Loading encodings from cache: '{settings.KNOWN_ENCODINGS_FILE}'")
        with open(settings.KNOWN_ENCODINGS_FILE, 'rb') as f:
            app_state.known_face_encodings = pickle.load(f)
        logger.info(f"Loaded {app_state.known_faces_count} known faces from cache.")
        return

    logger.info(f"No cache found. Loading faces from images in '{settings.SAVE_DIR}'")
    if not os.path.exists(settings.SAVE_DIR):
        os.makedirs(settings.SAVE_DIR)
        logger.info(f"Created directory: '{settings.SAVE_DIR}'")

    encodings = []
    for filename in os.listdir(settings.SAVE_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(settings.SAVE_DIR, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encs = face_recognition.face_encodings(image)
                if face_encs:
                    encodings.append(face_encs[0])
                else:
                    logger.warning(f"No face found in {filename}, skipping.")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}, skipping.")
    
    app_state.known_face_encodings = encodings
    
    with open(settings.KNOWN_ENCODINGS_FILE, 'wb') as f:
        pickle.dump(app_state.known_face_encodings, f)
    
    logger.info(f"Loaded and cached {app_state.known_faces_count} known faces.")

# --- Background Detection Task ---
async def _run_detection_loop(camera_index: int):
    global _camera_capture
    
    _camera_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not _camera_capture.isOpened():
        logger.error(f"Cannot open camera at index {camera_index}")
        return

    logger.info(f"Starting background detection loop on camera {camera_index}")
    frame_counter = 0
    last_processed_results: List[Tuple[tuple, str]] = []

    try:
        while not _stop_detection_flag.is_set():
            ret, frame = await asyncio.get_event_loop().run_in_executor(_executor, _camera_capture.read)
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame_counter += 1
            
            if frame_counter % settings.PROCESS_EVERY_N_FRAME == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = await asyncio.get_event_loop().run_in_executor(_executor, face_recognition.face_locations, rgb_small_frame)
                face_encodings = await asyncio.get_event_loop().run_in_executor(_executor, face_recognition.face_encodings, rgb_small_frame, face_locations)

                current_time = time.time()
                processed_results_this_frame = []

                expired_unknowns = [enc_tuple for enc_tuple, ts in app_state.recently_seen_unknowns.items() if current_time - ts > settings.RECENTLY_SEEN_TIMEOUT]
                for enc_tuple in expired_unknowns:
                    del app_state.recently_seen_unknowns[enc_tuple]

                if face_encodings:
                    for i, face_encoding in enumerate(face_encodings):
                        face_status = "UNKNOWN"
                        
                        is_known_face = False
                        if app_state.known_face_encodings:
                            matches = await asyncio.get_event_loop().run_in_executor(_executor, face_recognition.compare_faces, app_state.known_face_encodings, face_encoding, settings.DETECTION_THRESHOLD)
                            if True in matches:
                                is_known_face = True
                                face_status = "KNOWN"
                        
                        if not is_known_face:
                            is_recently_seen = False
                            if app_state.recently_seen_unknowns:
                                recent_encodings = [np.array(e) for e in app_state.recently_seen_unknowns.keys()]
                                matches = await asyncio.get_event_loop().run_in_executor(_executor, face_recognition.compare_faces, recent_encodings, face_encoding, settings.DETECTION_THRESHOLD)
                                if True in matches:
                                    is_recently_seen = True
                                    face_status = "PENDING"
                                    match_index = matches.index(True)
                                    matched_encoding_tuple = tuple(recent_encodings[match_index])
                                    app_state.recently_seen_unknowns[matched_encoding_tuple] = current_time

                            if not is_recently_seen:
                                face_status = "NEW"
                                logger.info(f"Detected new unique face! Total saved: {app_state.known_faces_count + 1}")
                                
                                app_state.known_face_encodings.append(face_encoding)
                                app_state.recently_seen_unknowns[tuple(face_encoding)] = current_time

                                top, right, bottom, left = face_locations[i]
                                orig_loc = (top * 2, right * 2, bottom * 2, left * 2)
                                await asyncio.get_event_loop().run_in_executor(_executor, save_new_face, frame.copy(), orig_loc)
                                _invalidate_encoding_cache()

                        location_on_original = tuple(x * 2 for x in face_locations[i])
                        processed_results_this_frame.append((location_on_original, face_status))
                
                last_processed_results = processed_results_this_frame

            color_map = {"KNOWN": (255, 100, 100), "NEW": (0, 0, 255), "PENDING": (0, 255, 255), "UNKNOWN": (0, 255, 0)}
            for (top, right, bottom, left), status in last_processed_results:
                color = color_map.get(status, (255,255,255))
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, status, (left, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, f"Total Faces: {app_state.known_faces_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pending Unknowns: {len(app_state.recently_seen_unknowns)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            app_state.frame_with_boxes = frame.copy()
            await asyncio.sleep(0.01)
            
    except asyncio.CancelledError:
        logger.info("Background detection task was cancelled.")
    finally:
        if _camera_capture and _camera_capture.isOpened():
            _camera_capture.release()
            logger.info("Camera released.")

# --- FastAPI App ---
app = FastAPI(title="Advanced Face Detection Service")

@app.on_event("startup")
async def startup_event():
    global _detection_task
    logger.info("Application startup...")
    load_known_faces_sync()
    _detection_task = asyncio.create_task(_run_detection_loop(camera_index=settings.CAMERA_INDEX))
    logger.info("Background detection task scheduled.")

@app.on_event("shutdown")
async def shutdown_event():
    global _detection_task
    logger.info("Application shutdown...")
    if _detection_task and not _detection_task.done():
        _stop_detection_flag.set()
        _detection_task.cancel()
        try:
            await _detection_task
        except asyncio.CancelledError:
            logger.info("Background task cancelled successfully.")
    _executor.shutdown(wait=True)
    logger.info("Application shut down completely.")

async def generate_video_frames():
    while not _stop_detection_flag.is_set():
        if app_state.frame_with_boxes is None:
            await asyncio.sleep(0.1)
            continue
        (flag, encodedImage) = cv2.imencode(".jpg", app_state.frame_with_boxes)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        await asyncio.sleep(1/30)

@app.get("/", summary="Main page with video stream")
async def main_page():
    html_content = """
    <html><head><title>Face Detection Stream</title>
    <style>
        body { font-family: sans-serif; background-color: #282c34; color: white; margin: 0; padding: 20px; text-align: center; } 
        h1 { color: #61dafb; } 
        #video-stream { border: 2px solid #61dafb; border-radius: 8px; max-width: 90%; height: auto; } 
        .info { margin-top: 20px; font-size: 1.2em; } 
        a { color: #61dafb; text-decoration: none; }
        a:hover { text-decoration: underline; }
        p { margin: 8px 0; }
    </style></head>
    <body>
        <h1>Live Face Detection Stream</h1>
        <img id="video-stream" src="/video_feed" alt="Video Stream">
        <div class="info">
            <p><strong>API Endpoints:</strong></p>
            <p><a href="/docs" target="_blank">/docs</a> &mdash; Interactive API Documentation</p>
            <p><a href="/status" target="_blank">/status</a> &mdash; Get current detection status</p>
            <p><a href="/faces" target="_blank">/faces</a> &mdash; List all saved face files</p>
        </div>
    </body></html>
    """
    return Response(content=html_content, media_type="text/html")

@app.get("/status", summary="Get current detection status details")
async def get_status():
    return {
        "total_unique_faces_saved": app_state.known_faces_count,
        "pending_unknown_faces_in_memory": len(app_state.recently_seen_unknowns),
        "processing_every_n_frame": settings.PROCESS_EVERY_N_FRAME,
        "recently_seen_timeout_seconds": settings.RECENTLY_SEEN_TIMEOUT,
    }

@app.get("/video_feed", summary="Stream video with detection overlays")
async def video_feed():
    return StreamingResponse(generate_video_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/faces", summary="List all saved face image files")
async def list_saved_faces():
    if not os.path.exists(settings.SAVE_DIR):
        return {"saved_faces": []}
    return {"saved_faces": sorted(os.listdir(settings.SAVE_DIR))}

@app.post("/faces/reload", summary="Force a reload of known faces from the directory")
async def reload_faces():
    _invalidate_encoding_cache()
    await asyncio.get_event_loop().run_in_executor(None, load_known_faces_sync)
    app_state.recently_seen_unknowns.clear()
    return {"message": "Reloading known faces completed.", "count": app_state.known_faces_count}

@app.delete("/faces/{filename}", summary="Delete a specific face file and update the database")
async def delete_face(filename: str):
    file_path = os.path.join(settings.SAVE_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    
    os.remove(file_path)
    logger.info(f"Deleted file: '{filename}'")
    
    _invalidate_encoding_cache()
    await asyncio.get_event_loop().run_in_executor(None, load_known_faces_sync)
    app_state.recently_seen_unknowns.clear()
    
    return {"message": f"'{filename}' deleted and faces reloaded.", "new_count": app_state.known_faces_count}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)