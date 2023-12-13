import os
import cv2
import pickle
import face_recognition
import argparse
import shutil
from tqdm import tqdm

# Load the reference encodings created in the script album.py
data = pickle.loads(open('face_encoding.pickle', "rb").read())

# Make the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# Provide a path to the directory containing test images and
# a path to the directory where you would like to save your output data
ap.add_argument("-i", "--test_directory", default='TestData',
                help="path to the test image directory")
ap.add_argument("-o", "--output_directory", default='OutputData',
                help="path to the output directory")
ap.add_argument("-t", "--tolerance", type=float, default=0.4,
                help="face recognition tolerance (default: 0.4)")
args = vars(ap.parse_args())

test_dir = args["test_directory"]
output_dir = args["output_directory"]
tolerance = args["tolerance"]

# Initialize a map linking the faces and the filenames they are found in the output
filemap = {names: [] for names in data["names"]}

# Get the total number of images in the directory for tqdm
total_images = len([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.heic'))])

# Loop over all the images in the test directory
for image_file in tqdm(os.listdir(test_dir), desc="Processing Images", unit="image", total=total_images):
    image_path = os.path.join(test_dir, image_file)

    # Read only image files & ignore the rest of the files
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.heic')):
        filename, file_extension = os.path.splitext(image_file)
    else:
        continue

    # Load the image
    test_image = cv2.imread(image_path)

    # Extract the position of the bounding box of the face and their corresponding face encodings
    bboxes = face_recognition.face_locations(test_image, model='hog')
    encodings = face_recognition.face_encodings(test_image, bboxes)

    names = []

    # Loop over the found encodings and compare them to the encodings in the reference database
    for encoding in encodings:

        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=tolerance)
        name = "Unknown"

        # If the test image has even a single face that matched a face in the database
        if True in matches:

            # Extract the matched indices
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Extract the corresponding names of the matched indices and get a vote count for each matched face name
            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # The name of the face with the maximum number of votes wins
            name = max(counts, key=counts.get)

        names.append(name)

    # Draw the bounding box around the faces with their detected names
    for ((top, right, bottom, left), name) in zip(bboxes, names):
        cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(test_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if name != "Unknown":
            filemap[name].append(filename)

            # Create a folder for each recognized name if not already created
            name_folder = os.path.join(output_dir, name)
            os.makedirs(name_folder, exist_ok=True)

            folder_name = name + "_recognized"
            recognized_faces_dir = os.path.join(output_dir, folder_name)
            os.makedirs(recognized_faces_dir, exist_ok=True)

            # Save the image to the respective folder
            output_path = os.path.join(name_folder, f"output_{filename}.jpg")
            cv2.imwrite(output_path, test_image)
            shutil.copy(image_path, recognized_faces_dir)

print("Action Performed Successfully")