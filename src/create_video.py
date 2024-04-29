import cv2
import os
from tqdm import tqdm
import logging

def create_video_from_images(image_folder, output_video_file, fps=30):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Checking if image folder exists.")
    if not os.path.exists(image_folder):
        logging.error("The specified image folder does not exist.")
        raise FileNotFoundError("The specified image folder does not exist.")
    
    logging.info("Listing and sorting images.")
    images = sorted(os.listdir(image_folder), key=lambda x: int(x.split('.')[0]) if x.endswith(".jpg") else 999999)
    images = [img for img in images if img.endswith(".jpg")]

    if not images:
        logging.error("No .jpg images found in the folder.")
        raise ValueError("No .jpg images found in the folder.")

    first_image_path = os.path.join(image_folder, images[0])
    logging.info(f"Reading the first image at {first_image_path}.")
    frame = cv2.imread(first_image_path)
    if frame is None:
        logging.error(f"Could not read the initial image at {first_image_path}.")
        raise FileNotFoundError(f"Could not read the initial image at {first_image_path}.")

    height, width, layers = frame.shape
    output_dir = os.path.dirname(output_video_file)
    if not os.path.exists(output_dir):
        logging.info("Output directory does not exist, creating now.")
        os.makedirs(output_dir)

    logging.info("Initializing video writer.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    frame_count = 0
    logging.info("Starting to write frames to video.")
    for image in tqdm(images, desc="Creating Video"):
        image_path = os.path.join(image_folder, image)
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Could not read image {image_path}. Skipping.")
        else:
            if img.shape != frame.shape:
                logging.warning(f"Image {image_path} is of a different size. Skipping.")
            else:
                video.write(img)
                frame_count += 1
