import os
import cv2
import numpy as np
import requests
import zipfile

# Configuration
dataset_url = "https://www.kaggle.com/datasets/cyx6666/coil20"  # Replace with the actual URL
dataset_zip_path = "coil-20.zip"
dataset_path = "coil-20"
processed_output_folder = "processed_panorama"

# Ensure the processed output folder exists
os.makedirs(processed_output_folder, exist_ok=True)

# Step 1: Download the dataset
def download_dataset(url, save_path):
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"Dataset downloaded to {save_path}.")

# Step 2: Extract the dataset
def extract_dataset(zip_path, extract_to):
    print(f"Extracting dataset from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset extracted to {extract_to}.")

# Step 3: Image processing and stitching function
def process_imgs(images):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    stitch_images = images[0]  # Initialize with the first image
    
    for i in range(0, len(images) - 1):  # Ensure we don't go out of bounds
        kp1, desc1 = sift.detectAndCompute(stitch_images, None)
        kp2, desc2 = sift.detectAndCompute(images[i+1], None)
        
        # Match descriptors
        matches = bf.match(desc1, desc2)
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        
        if len(sorted_matches) > 4:
            # construct the two sets of points
            src_p = np.float32([kp1[m.queryIdx].pt for m in sorted_matches]).reshape(-1, 1, 2)
            dst_p = np.float32([kp2[m.trainIdx].pt for m in sorted_matches]).reshape(-1, 1, 2)
            
            # Calculate the homography between the sets of points
            H, status = cv2.findHomography(src_p, dst_p, cv2.RANSAC, 4)
            
            if H is not None:
                width = stitch_images.shape[1] + images[i+1].shape[1]
                height = max(stitch_images.shape[0], images[i+1].shape[0])
                
                # Warp the current image using the homography matrix
                result = cv2.warpPerspective(stitch_images, H, (width, height))

                # Blending the overlapping region using weighted addition
                overlap = result[0:images[i+1].shape[0], 0:images[i+1].shape[1]]
                blended_region = cv2.addWeighted(overlap, 0.5, images[i+1], 0.5, 0)
                result[0:images[i+1].shape[0], 0:images[i+1].shape[1]] = blended_region

                # Update stitch_images with the new stitched result
                stitch_images = result
            else:
                print("Homography could not be found between images.")
        else:
            print(f"Not enough matches between images {i} and {i+1}")
        
    return stitch_images

# Main workflow
if not os.path.exists(dataset_path):
    download_dataset(dataset_url, dataset_zip_path)
    extract_dataset(dataset_zip_path, dataset_path)

# Process each object folder
for obj_num in range(1, 21):
    obj_folder = os.path.join(dataset_path, f'{obj_num}')
    
    if not os.path.exists(obj_folder):
        print(f"Folder {obj_folder} does not exist. Skipping...")
        continue
    
    image_files = sorted([f for f in os.listdir(obj_folder) if f.endswith('.png')])
    selected_images = image_files  # You can filter the selection here if needed
    selected_image_paths = [os.path.join(obj_folder, image) for image in selected_images]

    images = [cv2.imread(path) for path in selected_image_paths]

    # Process and stitch the images
    if len(images) > 0:
        panorama = process_imgs(images)

        # Save the final panorama
        output_path = os.path.join(processed_output_folder, f'panorama_obj_{obj_num}.jpg')
        cv2.imwrite(output_path, panorama)
        print(f"Saved panorama for object {obj_num} to {output_path}.")
