import face_recognition
import cv2
import os
import pickle
import numpy as np
import time
import sys
from datetime import datetime
import csv

# --- Configuration ---
ENCODINGS_FILE = "trained_faces.pkl"

# IMPORTANT: Only ONE of the INPUT_SOURCE lines below should be uncommented.
INPUT_SOURCE = 'webcam'

IMAGE_PATH = "test_image.jpg"
# VIDEO_PATH = "test_video.mp4"

# Adjusted for smoother, event-based detection
FRAME_SKIP_INTERVAL = 5
RECOGNITION_TOLERANCE = 0.45 # A slightly stricter tolerance for attendance

# --- New Configuration for Attendance System ---
ATTENDANCE_LOG_FILE = "attendance.csv"
LOG_DIR = "logs"
UNKNOWN_CAPTURES_DIR = "unknown_captures"

# Cooldown period in seconds for logging the same person again
LOG_COOLDOWN_SECONDS = 15
AWAY_TIMEOUT_SECONDS = 600 # Time in seconds a person can be away before being marked 'Absent' (10 minutes)
UNKNOWN_CAPTURE_COOLDOWN_SECONDS = 30 # Time in seconds to wait before capturing the same unknown face again

# Dictionaries to store the last time a person was seen and their attendance status
last_seen_time = {} # Format: {person_name: timestamp}
attendance_status = {} # Format: {person_name: 'Present' or 'Absent'}
last_unknown_capture_time = 0 # Stores the timestamp of the last unknown capture

# A simple counter for unique unknown faces in a session (resets on script restart)
unknown_face_counter = 0

def ensure_directories_exist():
    """Ensures that the log and unknown captures directories exist."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_CAPTURES_DIR, exist_ok=True)
    print(f"[INFO] Ensured '{LOG_DIR}' and '{UNKNOWN_CAPTURES_DIR}' directories exist.")

def initialize_attendance_log(known_names):
    """Initializes the CSV log file with a header if it doesn't exist."""
    attendance_file_path = os.path.join(LOG_DIR, ATTENDANCE_LOG_FILE)
    if not os.path.exists(attendance_file_path):
        with open(attendance_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Status', 'Timestamp'])
            print(f"[INFO] Created new attendance log file: '{ATTENDANCE_LOG_FILE}'")
    else:
        print(f"[INFO] Using existing attendance log file: '{ATTENDANCE_LOG_FILE}'")
    
    # Initialize attendance status for all known people to 'Absent'
    for name in known_names:
        attendance_status[name] = 'Absent'
        last_seen_time[name] = 0
    print(f"[INFO] Initialized attendance status for {len(known_names)} students.")

def update_attendance(person_name, status):
    """Updates the attendance status and logs the event to the CSV file."""
    if person_name not in attendance_status or attendance_status[person_name] != status:
        # Log the change only if the status is actually new
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance_file_path = os.path.join(LOG_DIR, ATTENDANCE_LOG_FILE)
        with open(attendance_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([person_name, status, timestamp])
        print(f"[ATTENDANCE] {person_name} marked as {status} at {timestamp}")
        attendance_status[person_name] = status


def capture_unknown_face(frame, face_location):
    """
    Captures and saves an image of an unknown face only once per cooldown period.
    """
    global unknown_face_counter
    global last_unknown_capture_time
    current_time = time.time()

    # Check the cooldown period before attempting to capture
    if current_time - last_unknown_capture_time < UNKNOWN_CAPTURE_COOLDOWN_SECONDS:
        return

    try:
        # Extract the face region from the frame
        top, right, bottom, left = face_location
        margin = 20
        top = max(0, top - margin)
        right = min(frame.shape[1], right + margin)
        bottom = min(frame.shape[0], bottom + margin)
        left = max(0, left - margin)
        face_image = frame[top:bottom, left:right]

        if face_image.size > 0:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(UNKNOWN_CAPTURES_DIR, f"unknown_{timestamp_str}_{unknown_face_counter}.jpg")
            cv2.imwrite(filename, face_image)
            print(f"[CAPTURE] Captured unknown face to: {filename}")
            unknown_face_counter += 1
            last_unknown_capture_time = current_time
        else:
            print("[WARNING] Captured face region was empty. Skipping save.")
    except Exception as e:
        print(f"[ERROR] An error occurred while attempting to capture and save an unknown face: {e}")


def load_encodings(file_path):
    """Loads known face encodings and names from a pickle file."""
    print(f"[INFO] Loading encodings from {file_path}...")
    if not os.path.exists(file_path):
        print(f"[ERROR] Encoded faces file '{file_path}' not found. "
              "Please run 'train.py' first to generate it.")
        return [], []
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]
    except Exception as e:
        print(f"[ERROR] Could not load encodings from {file_path}: {e}")
        return [], []

def main():
    ensure_directories_exist()
    known_face_encodings, known_face_names = load_encodings(ENCODINGS_FILE)

    if not known_face_encodings:
        print("[ERROR] Exiting: No known faces available for detection.")
        sys.exit(1)
    
    initialize_attendance_log(known_face_names)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam. Exiting.")
        sys.exit(1)

    print("[INFO] Starting webcam feed. Press 'q' to quit.")
    print("IMPORTANT: Click on the OpenCV window to make it active before pressing 'q' to quit.")

    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[INFO] End of video stream or failed to read frame.")
            break

        frame_count += 1
        processed_this_frame = (frame_count % FRAME_SKIP_INTERVAL == 0)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        face_encodings = []
        
        # Check for people who are now "Away"
        current_time = time.time()
        for name, last_seen in list(last_seen_time.items()):
            if last_seen > 0 and (current_time - last_seen) > AWAY_TIMEOUT_SECONDS:
                if attendance_status[name] == 'Present':
                    update_attendance(name, 'Absent')
                    
        if processed_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for i, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # If person is detected, update their status to 'Present' and their 'last seen' time
                    update_attendance(name, 'Present')
                    last_seen_time[name] = time.time()
                else:
                    # Capture unknown faces
                    top, right, bottom, left = face_locations[i]
                    original_face_location = (top * 4, right * 4, bottom * 4, left * 4)
                    capture_unknown_face(frame, original_face_location)

                # Display results on screen
                top, right, bottom, left = face_locations[i]
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                status_text = f"{name} ({attendance_status.get(name, 'Unknown')})"

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, status_text, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
        
        # Display overall attendance count
        present_count = list(attendance_status.values()).count('Present')
        absent_count = list(attendance_status.values()).count('Absent')
        cv2.putText(frame, f"Present: {present_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Absent: {absent_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Attendance System', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed. Exiting.")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Attendance process finished.")

if __name__ == "__main__":
    main()
