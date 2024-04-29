import cv2
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import random
from PIL import Image, ImageFilter



def convert_gt_to_csv(input_file, output_file):
    # Read the ground truth data
    df = pd.read_csv(input_file, header=None, names=['frame_id', 'object_id', 'x', 'y', 'width', 'height', 'unknown1', 'player/ball', 'unknown2'])
    
    # Write the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Converted data has been written to {output_file}")
    
def annotate_images(images_path, annotations_path, output_path):
    """
    Draw bounding boxes on all images based on the annotations provided in a CSV file
    and save them. Each image will have all corresponding annotations applied.

    Parameters:
    - images_path: Path to the directory containing the image files.
    - annotations_path: Path to the CSV file containing the annotations.
    - output_path: Path to the directory where annotated images will be saved.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the annotations
    annotations = pd.read_csv(annotations_path)

    # Get unique frame_ids from annotations
    unique_frame_ids = annotations['frame_id'].unique()

    # Iterate through each unique frame_id
    for frame_id in unique_frame_ids:
        # Construct the path to the image file with correct format
        image_file = os.path.join(images_path, f'{str(frame_id).zfill(6)}.jpg')

        # Load the image
        image = cv2.imread(image_file)
        if image is None:
            print(f'Image for frame_id {frame_id} not found in {images_path}.')
            continue  # Skip if the image file is not found

        # Filter annotations for the current frame_id
        frame_annotations = annotations[annotations['frame_id'] == frame_id]

        # Iterate through the annotations for the current frame_id, drawing bounding boxes
        for _, row in frame_annotations.iterrows():
            # Extract bounding box coordinates and draw the box
            x, y, width, height = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Draw green box

            # Optionally, add text for the object_id or any other annotation
            object_id = row['object_id']
            cv2.putText(image, str(object_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Save the annotated image
        output_file = os.path.join(output_path, f'annotated_{str(frame_id).zfill(6)}.jpg')
        cv2.imwrite(output_file, image)
        print(f'Annotated image saved as {output_file}')

    print('All images have been processed.')  # Indicate completion


def process_and_fill_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Find all unique object IDs
    unique_object_ids = data['object_id'].unique()
    
    # List to hold new data
    filled_data = []
    
    # Process each frame in the file
    for frame_id, group in tqdm(data.groupby('frame_id'), desc=f"Processing {file_path}"):
        present_ids = group['object_id'].unique()
        missing_ids = [obj_id for obj_id in unique_object_ids if obj_id not in present_ids]
        # Mark existing objects as in frame
        group['in_frame'] = 1
        filled_data.append(group)  # Existing data
        for missing_id in missing_ids:
            new_row = {
                'frame_id': frame_id, 
                'object_id': missing_id, 
                'x': np.nan, 'y': np.nan, 
                'width': np.nan, 'height': np.nan, 
                'unknown1': np.nan, 
                'player/ball': np.nan, 
                'unknown2': np.nan, 
                'in_frame': 0  # Mark as not in frame
            }
            filled_data.append(pd.DataFrame([new_row]))
    
    # Combine all the data
    new_data = pd.concat(filled_data, ignore_index=True)
    
    # Sort for easier visualization and consistency
    new_data = new_data.sort_values(by=['frame_id', 'object_id'])
    
    # Save the modified dataset back to a new file
    new_file_path = file_path.replace('.csv', '_with_in_frame.csv')
    new_data.to_csv(new_file_path, index=False)
    print(f"Processed and saved modified data to {new_file_path}")
    

def convert_to_yolo_format(csv_file_path, img_width, img_height, output_folder):
    """
    Converts data from a CSV file to YOLO format, considering the naming discrepancy
    between image files and label files.

    Parameters:
        csv_file_path (str): Path to the CSV file containing the data.
        img_width (int): Width of the images.
        img_height (int): Height of the images.
        output_folder (str): Folder to save the output files.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to collect annotations for each image
    annotations = {}
    
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header line
        
        for elements in reader:
            # Skip rows with missing or incomplete data
            if len(elements) < 8 or '' in elements[2:8]:
                continue  # Skip this row if any required fields are missing or empty
            
            # Extract data
            frame_id = elements[0]
            x, y, width, height = map(float, elements[2:6])
            class_id = int(float(elements[7])) - 1  # Adjust class ID to start from 0
            
            # Convert to YOLO format
            x_center, y_center = (x + width / 2) / img_width, (y + height / 2) / img_height
            width, height = width / img_width, height / img_height
            
            # Prepare text line in YOLO format
            yolo_format_line = f"{class_id} {x_center} {y_center} {width} {height}"
            
            # Format frame_id to match image filenames
            formatted_frame_id = str(frame_id).zfill(6)  # Pad frame_id with zeros to match image filename format
            
            # Add line to the corresponding frame's annotations
            if formatted_frame_id not in annotations:
                annotations[formatted_frame_id] = []
            annotations[formatted_frame_id].append(yolo_format_line)
    
    # Write annotations to files, one file per frame
    for frame_id, lines in annotations.items():
        output_file_path = os.path.join(output_folder, f"{frame_id}.txt")
        with open(output_file_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')

            
def split_dataset(images_path, labels_path, train_img_output, train_lbl_output, val_img_output, val_lbl_output, test_size):
    # Ensure output directories exist
    os.makedirs(train_img_output, exist_ok=True)
    os.makedirs(train_lbl_output, exist_ok=True)
    os.makedirs(val_img_output, exist_ok=True)
    os.makedirs(val_lbl_output, exist_ok=True)
    
    # List all image and label files
    images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    labels = [f for f in os.listdir(labels_path) if os.path.isfile(os.path.join(labels_path, f))]

    # Ensure matching names between images and labels
    base_names = [os.path.splitext(f)[0] for f in images]
    images, labels = zip(*[(f"{name}.jpg", f"{name}.txt") for name in base_names if f"{name}.txt" in labels])

    # Split into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    # Function to copy files
    def copy_files(files, source, destination):
        for file in files:
            shutil.copy(os.path.join(source, file), os.path.join(destination, file))

    # Copy files to their respective directories
    copy_files(train_images, images_path, train_img_output)
    copy_files(train_labels, labels_path, train_lbl_output)
    copy_files(val_images, images_path, val_img_output)
    copy_files(val_labels, labels_path, val_lbl_output)
    

def split_annotations(original_labels_dir, labels_ball_dir, labels_player_dir):
    # Ensure output directories exist
    os.makedirs(labels_ball_dir, exist_ok=True)
    os.makedirs(labels_player_dir, exist_ok=True)

    # List all annotation files
    files = [f for f in os.listdir(original_labels_dir) if f.endswith('.txt')]
    
    for file in files:
        path = os.path.join(original_labels_dir, file)
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Process annotations
        # For ball annotations, preserve class ID as 0
        ball_lines = [line for line in lines if int(line.split()[0]) == 0]
        # For player annotations, change class ID to 0
        player_lines = ['0' + line[line.index(' '):] for line in lines if int(line.split()[0]) == 1]
        
        # Write separate annotation files for balls and players
        if ball_lines:
            with open(os.path.join(labels_ball_dir, file), 'w') as f:
                f.writelines(ball_lines)
        if player_lines:
            with open(os.path.join(labels_player_dir, file), 'w') as f:
                f.writelines(player_lines)
                

def copy_and_rename_images_labels(image_dirs, label_dirs, output_image_dir, output_label_dir, starting_index=1):
    """
    Copies and renames images and their corresponding label files from multiple directories into single output directories.
    
    Parameters:
    - image_dirs: List of directories containing images.
    - label_dirs: List of directories containing corresponding label files.
    - output_image_dir: Output directory for images.
    - output_label_dir: Output directory for label files.
    - starting_index: Starting index for renaming files.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    current_index = starting_index
    
    for image_dir, label_dir in zip(image_dirs, label_dirs):
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        
        for image_file, label_file in zip(image_files, label_files):
            # Generate new file names
            new_image_name = f"{str(current_index).zfill(6)}.jpg"
            new_label_name = f"{str(current_index).zfill(6)}.txt"
            
            # Copy and rename images
            shutil.copy(os.path.join(image_dir, image_file), os.path.join(output_image_dir, new_image_name))
            # Copy and rename labels
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_label_dir, new_label_name))
            
            current_index += 1


def draw_yolo_annotations(image_path, annotation_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Read annotations
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.split())

        # Convert relative coordinates to absolute coordinates
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
        x_min, y_min = int(x_center - w / 2), int(y_center - h / 2)
        x_max, y_max = int(x_center + w / 2), int(y_center + h / 2)

        # Draw rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Optionally, label the rectangle with the class_id
        cv2.putText(img, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Save or display the image
    cv2.imwrite(output_path, img)
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
def process_images_every_n(image_dir, label_dir, output_dir, n=100):
    os.makedirs(output_dir, exist_ok=True)
    images = sorted([img for img in os.listdir(image_dir) if img.endswith('.jpg')])
    selected_images = images[::n]  # Select every 100th image

    for img_name in selected_images:
        image_path = os.path.join(image_dir, img_name)
        annotation_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        output_path = os.path.join(output_dir, img_name)

        # Check if the annotation file exists
        if os.path.exists(annotation_path):
            draw_yolo_annotations(image_path, annotation_path, output_path)
        else:
            print(f"Annotation file for {img_name} does not exist.")
            

def copy_images(input_dir, output_dir, extensions=['.jpg', '.jpeg', '.png']):
    """
    Copies all image files from the input directory to the output directory.
    Only copies files with the specified extensions.

    Parameters:
    - input_dir (str): Path to the input directory containing images to copy.
    - output_dir (str): Path to the output directory where images will be copied.
    - extensions (list): List of file extensions to copy.
    """
    # Check if the output directory exists, create it if not
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in the input directory
    files = os.listdir(input_dir)
    
    # Copy each file to the output directory if it matches the extensions
    for file in files:
        if any(file.lower().endswith(ext) for ext in extensions):
            shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir, file))
            

import os
import random
from PIL import Image, ImageFilter

def process_images(image_folder, label_folder, output_percentage=0.05, expand_pct=0.1, blur_radius=100):
    # List all files in the image and label folders
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    # Calculate the number of images to process
    total_images = len(image_files)
    num_to_process = int(total_images * output_percentage)

    # Randomly select a subset of image files to process
    selected_images = random.sample(image_files, num_to_process)

    # Process each selected image
    for image_file in selected_images:
        # Construct full paths for the image and label files
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Read the image
        with Image.open(image_path) as img:
            # Check if the corresponding label file exists
            if os.path.exists(label_path):
                # Read the label file to get regions to blur
                with open(label_path, 'r') as file:
                    lines = file.readlines()

                # Blur regions specified in the label file
                for line in lines:
                    # Parse the line to extract coordinates (YOLOv8 format)
                    parts = line.strip().split()
                    # Typically: class cx cy w h (normalized)
                    # Compute absolute coordinates (assuming top-left origin)
                    cx, cy, w, h = map(float, parts[1:5])
                    x = int((cx - w / 2) * img.width)
                    y = int((cy - h / 2) * img.height)
                    w = int(w * img.width)
                    h = int(h * img.height)

                    # Expand the region by a specified percentage
                    expand_w = int(w * expand_pct)
                    expand_h = int(h * expand_pct)
                    box = (max(0, x - expand_w), max(0, y - expand_h), min(img.width, x + w + expand_w), min(img.height, y + h + expand_h))

                    # Apply a strong blur to the expanded region
                    region = img.crop(box)
                    blurred_region = region.filter(ImageFilter.GaussianBlur(blur_radius))
                    img.paste(blurred_region, box)

            # Save the processed image back to the same folder with a modified name
            img.save(os.path.join(image_folder, 'processed_' + image_file))

            # Create a new empty label file for the processed image
            new_label_name = 'processed_' + image_file.replace('.jpg', '.txt').replace('.png', '.txt')
            with open(os.path.join(label_folder, new_label_name), 'w') as new_label:
                new_label.write("")  # Writing an empty file


def change_class_id(folder_path):
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Check if the file is a .txt file
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            modified_lines = []
            for line in lines:
                parts = line.split()
                if parts:  # Check if the line is not empty
                    class_id = int(parts[0])
                    if class_id > 0:
                        parts[0] = '1'  # Change class to 1 if it is greater than 0
                    modified_line = ' '.join(parts)
                    modified_lines.append(modified_line + '\n')
            
            # Write the modified lines back to the file
            with open(filepath, 'w') as file:
                file.writelines(modified_lines)

