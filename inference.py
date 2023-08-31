import cv2
import pickle
import mediapipe as mp
from deepface import DeepFace
from deepface.commons import distance as dst

# Load the precomputed employee embeddings dictionary
with open("employee_embeddings.pkl", "rb") as f:
    employee_embeddings = pickle.load(f)


# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize the live camera
camera = cv2.VideoCapture(0)
print("Press 'q' to Quit")
while True:
    ret, frame = camera.read()
    
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
                        #print(distance)
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
    
    # Display the frame with recognized names
    cv2.imshow("Smart Attendance System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
