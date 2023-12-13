import os
import face_recognition
import pickle

# Provide the path to the directory containing face images for training
train_dir = 'data/nayan_train'

# Initialize lists to store face encodings and corresponding names
encodings = []
names = []

# Loop over all the images in the training directory
for count, person_folder in enumerate(os.listdir(train_dir)):
    person_path = os.path.join(train_dir, person_folder)

    # Ensure that it's a directory
    if os.path.isdir(person_path):
        print(f"Processing person {count + 1} of {len(os.listdir(train_dir))}. Person: {person_folder}")

        # Loop over all the images for the current person
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)

            # Read only image files & ignore the rest of the files
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.heic')):
                print(f"Processing image: {image_file}")
            else:
                print(f"Skipping non-image file: {image_file}")
                continue

            # Load the image
            image = face_recognition.load_image_file(image_path)

            # Find all face locations and face encodings in the image
            face_locations = face_recognition.face_locations(image)
            face_encoding = face_recognition.face_encodings(image, face_locations)

            if len(face_encoding) > 0:
                # Assuming only one face is present in the image for simplicity
                encodings.append(face_encoding[0])
                names.append(person_folder)

# Create a dictionary to store encodings and corresponding names
data = {"encodings": encodings, "names": names}

# Save the encodings to a pickle file
output_pickle_path = 'face_encoding.pickle'
with open(output_pickle_path, 'wb') as f:
    pickle.dump(data, f)

print(f"Encodings saved to {output_pickle_path}")