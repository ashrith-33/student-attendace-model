import face_recognition
import os
import pickle

dataset_path = "dataset"
known_encodings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):         
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        print(f"[INFO] Processing {image_path}")

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"[WARNING] No face found in {image_path}")

data = {"encodings": known_encodings, "names": known_names}
with open("trained_faces.pkl", "wb") as f:
    pickle.dump(data, f)

print(" Training complete! Saved to 'trained_faces.pkl'")