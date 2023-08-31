import os
import pickle
from deepface import DeepFace

# Path to the folder containing employee images
employee_images_path = "data"

# Create a dictionary to store employee names and their corresponding embeddings
employee_embeddings = {}

# Load the pre-trained deep learning model
#model = DeepFace.build_model("Facenet")

# Iterate through the employee image folders
print('Traning Started ... %')
for employee_name in os.listdir(employee_images_path):
    employee_folder = os.path.join(employee_images_path, employee_name)
    embeddings = []
    for image_file in os.listdir(employee_folder):
        image_path = os.path.join(employee_folder, image_file)
        #face = DeepFace.detectFace(image_path, detector_backend='opencv')[0]  # Assuming one face per image
        embedding = DeepFace.represent(image_path, model_name = "Facenet",enforce_detection= False)
        embeddings.append(embedding)
    employee_embeddings[employee_name] = embeddings
    
transformed_data = {}
for person, embeddings_list in employee_embeddings.items():
    transformed_data[person] = [item['embedding'] for sublist in embeddings_list for item in sublist]
# Save the employee embeddings dictionary to a file
with open("employee_embeddings.pkl", "wb") as f:
    pickle.dump(transformed_data, f)
print('Traning Finished')
