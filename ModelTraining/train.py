import os
import cv2
import face_recognition
import pickle
from tqdm import tqdm

# Make the argument parser and parse the arguments
import argparse
ap = argparse.ArgumentParser()

# Provide a path to the directory containing training images and
# a path to the file where you would like to save your trained model
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input training image directory")
ap.add_argument("-o", "--output_file", required=True,
                help="path to the output trained model file (face_encoding.pickle)")
ap.add_argument("-t", "--tolerance", type=float, default=0.5,
                help="face recognition tolerance (default: 0.5)")
args = vars(ap.parse_args())

input_dir = args["input_directory"]
output_file = args["output_file"]
tolerance = args["tolerance"]

# Initialize lists to store face encodings and corresponding names
encodings = []
names = []

# Loop over all the images in the training directory
for person_folder in tqdm(os.listdir(input_dir), desc="Processing Persons", unit="person"):
    person_path = os.path.join(input_dir, person_folder)

    # Check if the item in the directory is a folder (person's name)
    if os.path.isdir(person_path):

        # Loop over all the images in the person's directory
        for image_file in os.listdir(person_path):

            image_path = os.path.join(person_path, image_file)

            # Read only image files & ignore the rest of the files
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.heic')):

                # Load the image
                image = face_recognition.load_image_file(image_path)

                # Extract the face encodings from the image
                encoding = face_recognition.face_encodings(image, model="cnn")

                # Check if a face is detected and encoded
                if len(encoding) > 0:
                    encodings.append(encoding[0])  # Assuming only one face is in the image
                    names.append(person_folder)

# Create a dictionary to store the face encodings, corresponding names, and tolerance
data = {"encodings": encodings, "names": names, "tolerance": tolerance}

# Save the face encodings, names, and tolerance to a pickle file
with open(output_file, "wb") as file:
    pickle.dump(data, file)

print(f"Face recognition model trained successfully and saved to {output_file} with tolerance {tolerance}")
