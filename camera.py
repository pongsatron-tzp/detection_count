import cv2
import time
import face_recognition
import numpy as np
import os
import shutil

SAVE_DIR = "detected_faces"

known_face_encodings = []
known_faces_count = 0
last_detection_time = {}

# New global variables for tracking last saved person
last_saved_person_encoding = None
last_saved_person_cooldown_active = False
last_saved_person_seen_time = 0
COOLDOWN_TIMEOUT_IF_NOT_SEEN = 5 # seconds
MIN_COOLDOWN_BEFORE_EXIT_CHECK = 0.5 # seconds


def setup_save_directory():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

# def cleanup_save_directory():
#     if os.path.exists(SAVE_DIR):
#         print(f"Deleting all image files in '{SAVE_DIR}'...")
#         shutil.rmtree(SAVE_DIR)
#         print("Files deleted.")

def load_known_faces():
    global known_face_encodings, known_faces_count
    known_face_encodings = []
    known_faces_count = 0
    print(f"Loading known faces from '{SAVE_DIR}'...")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for filename in os.listdir(SAVE_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(SAVE_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_faces_count += 1
                print(f"Loaded: {filename}")
            else:
                print(f"No face found in {filename}")
    print(f"Total faces loaded: {known_faces_count}")


def detect_face_from_camera(camera_index=0, detection_threshold=0.5):
    global known_face_encodings, known_faces_count, last_detection_time
    global last_saved_person_encoding, last_saved_person_cooldown_active, last_saved_person_seen_time

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return

    setup_save_directory()
    load_known_faces()

    print(f"Opening camera at index {camera_index}... Press 'q' to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not receive frame from camera.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            status_text = "No Face Detected"
            status_color = (0, 0, 255)
            
            last_saved_person_is_in_frame = False 
            
            if len(face_encodings) > 0:
                current_time = time.time()
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    x, y, w, h = left, top, right - left, bottom - top
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    match_found = False
                    min_distance = float('inf')

                    if len(known_face_encodings) > 0:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        matches = face_distances <= detection_threshold
                        
                        if True in matches:
                            match_found = True
                            best_match_index = np.argmin(face_distances)
                            min_distance = face_distances[best_match_index]
                            similarity_percent = ((1 - min_distance) * 100)
                            status_text = f"Known Person: (Similarity: {similarity_percent:.2f}%)"
                            status_color = (0, 255, 0)
                            
                            if last_saved_person_cooldown_active and last_saved_person_encoding is not None:
                                if face_recognition.compare_faces([last_saved_person_encoding], face_encoding, tolerance=detection_threshold)[0]:
                                    last_saved_person_is_in_frame = True
                                    last_saved_person_seen_time = current_time
                                    
                    if not match_found:
                        if not last_saved_person_cooldown_active:
                            known_face_encodings.append(face_encoding)
                            known_faces_count += 1
                            
                            timestamp = int(current_time)
                            face_image_path = os.path.join(SAVE_DIR, f"person_{known_faces_count}_{timestamp}.jpg")
                            face_img = frame[y:y+h, x:x+w]
                            cv2.imwrite(face_image_path, face_img)
                            print(f"Saved new person's image: {face_image_path}")
                            
                            status_text = f"New Person: Detected and saved person {known_faces_count}!"
                            status_color = (0, 165, 255)
                            
                            last_saved_person_encoding = face_encoding
                            last_saved_person_cooldown_active = True
                            last_saved_person_seen_time = current_time
                            
                        else:
                            status_text = f"Waiting for prev. person to leave..."
                            status_color = (0, 200, 200)
                    
                    break # Process only the first face for display logic

            if last_saved_person_cooldown_active:
                if not last_saved_person_is_in_frame:
                    if current_time - last_saved_person_seen_time > MIN_COOLDOWN_BEFORE_EXIT_CHECK:
                        last_saved_person_cooldown_active = False
                        last_saved_person_encoding = None
                        print("Last saved person has left the frame. Cooldown reset.")
                        status_text = "Ready for new person."
                        status_color = (0, 255, 0)
                elif current_time - last_saved_person_seen_time > COOLDOWN_TIMEOUT_IF_NOT_SEEN:
                    last_saved_person_cooldown_active = False
                    last_saved_person_encoding = None
                    print("Last saved person timed out (not seen for too long). Cooldown reset.")
                    status_text = "Ready (Timeout)."
                    status_color = (0, 255, 0)
            
            display_text = f"{status_text} | Unique Persons: {known_faces_count}"
            cv2.putText(frame, display_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

            cv2.imshow('Face Recognition Status', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed and program ended.")
        # cleanup_save_directory()

if __name__ == "__main__":
    detect_face_from_camera(camera_index=1)