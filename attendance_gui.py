import cv2
import pickle
import mediapipe as mp
from deepface import DeepFace
from deepface.commons import distance as dst
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load the precomputed employee embeddings dictionary
with open("employee_embeddings.pkl", "rb") as f:
    employee_embeddings = pickle.load(f)

# Initialize the live camera
camera = cv2.VideoCapture(0)

# Create a Tkinter window
window = tk.Tk()
window.title("Smart Attendance System")
window.bind('<q>', lambda e: window.destroy())

# Create a label to display attendance status
attendance_label = Label(window, text="", font=("Helvetica", 16))
attendance_label.pack()

# Create a label to display the video feed
video_label = tk.Label(window)
video_label.pack()

# List to store attended people
attended_people = []

def update_attended_history():
    attended_text = "\n".join(attended_people)
    attended_history.config(text=attended_text)
    window.after(1000, update_attended_history)

# Create a label to display attended history
attended_history_label = Label(window, text="Attended People:", font=("Helvetica", 16))
attended_history_label.pack(pady=10)

# Create a label to display attended history
attended_history = Label(window, text="", font=("Helvetica", 14))
attended_history.pack()
print("Press 'q' to Quit")
while True:
    ret, frame = camera.read()

    # ... (rest of the face recognition code remains the same)
    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using Mediapipe
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Crop the face area
            cropped_face = frame[y-40:y+h+5, x-40:x+w+5]
            
            # Get the embedding of the cropped face
            try:
                face_embedding = DeepFace.represent(cropped_face, model_name="Facenet", enforce_detection=False)
            except: 
                pass
            # Compare the face embedding with the employee embeddings
            min_distance = 6
            recognized_employee = "Unknown"
            
            for employee_name, embedding_list in employee_embeddings.items():
                for employee_embedding in embedding_list :
                    try: 
                        distance = dst.findEuclideanDistance(face_embedding[0]['embedding'],
                                                          employee_embedding)
                    except:
                        pass
                    if distance < min_distance:
                        min_distance = distance
                        recognized_employee = employee_name
            
            # Draw bounding box and recognized name on the frame
            if recognized_employee == "Unknown" :
                cv2.rectangle(frame, (x-40, y-40), (x + w + 5, y + h + 5), (0, 0, 255), 2)
                cv2.rectangle(frame, (x-40, y-70), (x+100, y-40), (0, 0, 255), -1)
                cv2.putText(frame, recognized_employee, (x-20, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else: 
                cv2.rectangle(frame, (x-40, y-40), (x + w + 5, y + h + 5), (0, 255, 0), 2)
                cv2.rectangle(frame, (x-40, y-70), (x+100, y-40), (0, 255, 0), -1)
                cv2.putText(frame, recognized_employee, (x-20, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
            # Convert the OpenCV image to a format compatible with Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2image)
            tk_image = ImageTk.PhotoImage(image=pil_image)

            # Update the label with the attendance status
            if recognized_employee != "Unknown":
                if recognized_employee not in attended_people:
                    attended_people.append(recognized_employee)
                    update_attended_history()
                attendance_label.config(text=f"{recognized_employee} - Attended", fg="green")
            else:
                attendance_label.config(text="Unknown", fg="red")

    # Update the video label with the new image
    video_label.config(image=tk_image)
    video_label.image = tk_image

    # Update the GUI
    window.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
