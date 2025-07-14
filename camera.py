import cv2
import time
import face_recognition
import numpy as np
import os
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn

# --- Global Configuration and State ---
SAVE_DIR = "detected_faces"
# สถานะที่ใช้ร่วมกันระหว่าง FastAPI app และ Background task
current_known_face_encodings = []
current_known_faces_count = 0
current_last_saved_person_encoding = None
current_last_saved_person_cooldown_active = False
current_last_saved_person_seen_time = 0
# [สตรีมมิ่ง] ตัวแปรสำหรับเก็บเฟรมล่าสุดเพื่อสตรีม
current_frame_with_boxes = None
# การตั้งค่า Cooldown และ Threshold
COOLDOWN_TIMEOUT_IF_NOT_SEEN = 3
DETECTION_THRESHOLD = 0.73 # ค่านี้ค่อนข้างผ่อนปรน อาจทำให้ทักคนผิดได้ง่าย
# ตัวควบคุมสำหรับ Background task
_camera_capture = None
_detection_task = None
_stop_detection_flag = asyncio.Event()
_executor = ThreadPoolExecutor(max_workers=2)

# [บันทึกภาพ] ฟังก์ชันใหม่สำหรับบันทึกใบหน้า
def save_new_face(frame, face_location):
    """
    ตัดภาพเฉพาะส่วนใบหน้าจากเฟรมและบันทึกเป็นไฟล์ .jpg
    """
    try:
        # ดึงตำแหน่งของใบหน้า (top, right, bottom, left)
        top, right, bottom, left = face_location
        
        # ขยายกรอบเล็กน้อยเพื่อให้ได้ภาพใบหน้าที่สมบูรณ์ขึ้น (เผื่อผมหรือคาง)
        padding = 30
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(frame.shape[0], bottom + padding)
        right = min(frame.shape[1], right + padding)
        
        # ตัดภาพใบหน้า (crop)
        face_image = frame[top:bottom, left:right]

        # สร้างชื่อไฟล์ที่ไม่ซ้ำกันโดยใช้วันที่และเวลา
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(SAVE_DIR, f"face_{timestamp}.jpg")
        
        # ตรวจสอบว่าภาพที่ตัดมาไม่เล็กเกินไป
        if face_image.size == 0:
            print(f"[บันทึกภาพ] คำเตือน: ไม่สามารถบันทึกภาพได้เนื่องจากขนาดภาพเป็น 0 (อาจจะอยู่นอกขอบเฟรม)")
            return

        # บันทึกไฟล์ภาพ
        cv2.imwrite(filename, face_image)
        print(f"[บันทึกภาพ] บันทึกใบหน้าใหม่เรียบร้อยแล้วที่: {filename}")
        
    except Exception as e:
        print(f"[บันทึกภาพ] เกิดข้อผิดพลาดในการบันทึกภาพใบหน้า: {e}")

# --- Helper Functions ---
def setup_save_directory():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"[Setup] สร้างไดเรกทอรี: '{SAVE_DIR}'")

def load_known_faces_sync():
    global current_known_face_encodings, current_known_faces_count
    current_known_face_encodings = []
    print(f"[Setup] กำลังโหลดใบหน้าที่รู้จักจาก '{SAVE_DIR}'...")
    setup_save_directory()
    for filename in os.listdir(SAVE_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(SAVE_DIR, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    current_known_face_encodings.append(encodings[0])
                else:
                    print(f"[Setup] คำเตือน: ไม่พบใบหน้าใน {filename} ข้ามไป")
            except Exception as e:
                print(f"[Setup] ข้อผิดพลาดในการโหลด {filename}: {e} ข้ามไป")
    current_known_faces_count = len(current_known_face_encodings)
    print(f"[Setup] โหลดใบหน้าที่รู้จักทั้งหมด: {current_known_faces_count}")

# --- Background Detection Task (แก้ไขเพื่อเรียกใช้ฟังก์ชันบันทึก) ---
async def _run_detection_loop(
    camera_index: int,
    stop_time_str: str = None,
    run_duration_seconds: int = None
):
    global _camera_capture, current_frame_with_boxes
    global current_known_face_encodings, current_known_faces_count
    global current_last_saved_person_encoding, current_last_saved_person_cooldown_active, current_last_saved_person_seen_time
    
    _camera_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not _camera_capture.isOpened():
        print(f"[Camera] ข้อผิดพลาด: ไม่สามารถเปิดกล้องที่ดัชนี {camera_index}")
        return
    print(f"[Service] กำลังเริ่มต้น Background face detection loop (กล้อง: {camera_index})...")

    try:
        while not _stop_detection_flag.is_set():
            ret, frame = await asyncio.get_event_loop().run_in_executor(_executor, _camera_capture.read)
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = await asyncio.get_event_loop().run_in_executor(_executor, face_recognition.face_locations, rgb_small_frame)
            face_encodings = await asyncio.get_event_loop().run_in_executor(_executor, face_recognition.face_encodings, rgb_small_frame, face_locations)
            
            current_time = time.time()
            last_saved_person_is_in_frame = False
            
            if len(face_encodings) > 0:
                # [บันทึกภาพ] ใช้ enumerate เพื่อให้ได้ index มาจับคู่กับ face_locations
                for i, face_encoding in enumerate(face_encodings):
                    is_known_face = False
                    if len(current_known_face_encodings) > 0:
                        matches = await asyncio.get_event_loop().run_in_executor(_executor, face_recognition.compare_faces, current_known_face_encodings, face_encoding, DETECTION_THRESHOLD)
                        if True in matches:
                            is_known_face = True
                    
                    if current_last_saved_person_cooldown_active and current_last_saved_person_encoding is not None:
                        is_last_saved_person_arr = await asyncio.get_event_loop().run_in_executor(_executor, face_recognition.compare_faces, [current_last_saved_person_encoding], face_encoding, DETECTION_THRESHOLD)
                        if is_last_saved_person_arr[0]:
                            last_saved_person_is_in_frame = True
                            current_last_saved_person_seen_time = current_time
                    
                    if not is_known_face:
                        if not current_last_saved_person_cooldown_active:
                            current_known_face_encodings.append(face_encoding)
                            current_known_faces_count += 1
                            print(f"[Detection] ตรวจพบบุคคลใหม่ที่ไม่ซ้ำกัน! บุคคลทั้งหมด: {current_known_faces_count}")
                            
                            current_last_saved_person_encoding = face_encoding
                            current_last_saved_person_cooldown_active = True
                            current_last_saved_person_seen_time = current_time
                            last_saved_person_is_in_frame = True

                            # [บันทึกภาพ] เรียกใช้ฟังก์ชันบันทึกภาพตรงนี้
                            # เราต้องใช้ตำแหน่งใบหน้าจาก face_locations ที่สอดคล้องกัน
                            # และขยายตำแหน่งกลับไปเป็นขนาดของเฟรมเต็ม
                            face_location_on_small_frame = face_locations[i]
                            top, right, bottom, left = face_location_on_small_frame
                            location_on_original_frame = (top * 2, right * 2, bottom * 2, left * 2)
                            
                            # เรียกใช้ฟังก์ชันบันทึกใน ThreadPoolExecutor เพื่อไม่ให้ block loop หลัก
                            await asyncio.get_event_loop().run_in_executor(
                                _executor,
                                save_new_face,
                                frame.copy(), # ส่งสำเนาของเฟรมเต็มไป
                                location_on_original_frame
                            )
            
            if current_last_saved_person_cooldown_active:
                if not last_saved_person_is_in_frame:
                    if current_time - current_last_saved_person_seen_time > COOLDOWN_TIMEOUT_IF_NOT_SEEN:
                        print(f"[Detection] Cooldown หมดเวลา... พร้อมใช้งานอีกครั้ง")
                        current_last_saved_person_cooldown_active = False
                        current_last_saved_person_encoding = None

            # ... (ส่วนวาดภาพและอัปเดต buffer เหมือนเดิม) ...
            for (top, right, bottom, left) in face_locations:
                top *= 2; right *= 2; bottom *= 2; left *= 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            status_text = f"Total Unique Faces: {current_known_faces_count}"
            cooldown_status = "COOLDOWN ACTIVE" if current_last_saved_person_cooldown_active else "Ready to detect new"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, cooldown_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            current_frame_with_boxes = frame.copy()
            
            await asyncio.sleep(0.01)
            
    except asyncio.CancelledError:
        print("[Service] Background detection task ถูกยกเลิก")
    finally:
        if _camera_capture and _camera_capture.isOpened():
            _camera_capture.release()
            print("[Camera] กล้องถูกปล่อยแล้ว")
        _executor.shutdown(wait=True)
        print("[Service] Background detection loop หยุดทำงาน")

# --- (ส่วนที่เหลือของไฟล์ตั้งแต่ generate_video_frames() ลงไปเหมือนเดิมทั้งหมด) ---

# [สตรีมมิ่ง] ฟังก์ชัน Generator สำหรับสร้างเฟรมวิดีโอ
async def generate_video_frames():
    while not _stop_detection_flag.is_set():
        if current_frame_with_boxes is None:
            await asyncio.sleep(0.1)
            continue
        (flag, encodedImage) = cv2.imencode(".jpg", current_frame_with_boxes)
        if not flag:
            await asyncio.sleep(0.1)
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        await asyncio.sleep(1/30)

# --- FastAPI App ---
app = FastAPI(title="Face Counting and Streaming Service")
CAMERA_INDEX = 1
SERVICE_STOP_TIME = None
SERVICE_RUN_DURATION = None

@app.on_event("startup")
async def startup_event():
    global _detection_task
    print("[App] แอปพลิเคชันเริ่มต้น...")
    load_known_faces_sync()
    _detection_task = asyncio.create_task(
        _run_detection_loop(camera_index=CAMERA_INDEX)
    )
    print("[App] Background detection task กำหนดเวลาแล้ว")

@app.on_event("shutdown")
async def shutdown_event():
    global _detection_task, _stop_detection_flag
    print("[App] แอปพลิเคชันกำลังปิด...")
    if _detection_task and not _detection_task.done():
        _stop_detection_flag.set()
        _detection_task.cancel()
        try:
            await _detection_task
        except asyncio.CancelledError:
            print("[App] Background task ถูกยกเลิกเรียบร้อย")
    print("[App] แอปพลิเคชันปิดตัวลงสมบูรณ์")

# --- API Endpoints ---
@app.get("/", summary="หน้าหลักพร้อมวิดีโอสตรีม")
async def main_page():
    html_content = """
    <html><head><title>Face Detection Stream</title><style>body { font-family: sans-serif; background-color: #282c34; color: white; margin: 0; padding: 20px; text-align: center; } h1 { color: #61dafb; } #video-stream { border: 2px solid #61dafb; border-radius: 8px; max-width: 90%; height: auto; } .info { margin-top: 20px; font-size: 1.2em; }</style></head>
    <body><h1>Live Face Detection Stream</h1><img id="video-stream" src="/video_feed" alt="Video Stream"><div class="info"><p>API Endpoints:</p><p><a href="/docs" target="_blank">/docs</a> - Interactive API Documentation</p><p><a href="/face_count" target="_blank">/face_count</a> - Get Current Unique Face Count</p></div></body></html>
    """
    return Response(content=html_content, media_type="text/html")

@app.get("/face_count", response_model=int, summary="รับจำนวนใบหน้าที่ไม่ซ้ำกันในปัจจุบัน")
async def get_face_count():
    return current_known_faces_count

@app.get("/status", summary="รับรายละเอียดสถานะการตรวจจับปัจจุบัน")
async def get_status():
    return {"total_unique_faces_detected": current_known_faces_count, "cooldown_active": current_last_saved_person_cooldown_active}

@app.get("/video_feed", summary="สตรีมวิดีโอพร้อมแสดงผลการตรวจจับ")
async def video_feed():
    return StreamingResponse(generate_video_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)