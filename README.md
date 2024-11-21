
# SIFT Image Stitching Using COIL-20 Dataset

This project implements the Scale-Invariant Feature Transform (SIFT) algorithm to stitch images together using the COIL-20 dataset. The goal is to create a panorama for each object in the dataset by combining 15 images taken from different angles.

## Dataset Information

The COIL-20 dataset consists of grayscale images of 20 different objects, each captured from 72 different angles (spaced at 5-degree intervals). This dataset is commonly used for object recognition and computer vision tasks.

Dataset Link: [COIL-20 Dataset on Kaggle](https://www.kaggle.com/datasets/cyx6666/coil20)

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Requests

Install the necessary Python libraries using:

```bash
pip install opencv-python-headless numpy requests
```

## How It Works

1. **Download Dataset**: The script downloads the dataset if it's not already present locally.
2. **Extract Dataset**: Extracts the COIL-20 dataset from a ZIP file.
3. **Feature Matching with SIFT**: Detects keypoints and descriptors using the SIFT algorithm and matches features between images.
4. **Homography & Warping**: Computes homography matrices for alignment and warps images into a stitched panorama.
5. **Panorama Creation**: Blends the overlapping regions to create smooth transitions between images.

## Running the Code

1. Download or clone this repository.
2. Place the `coil-20.zip` dataset in the project folder or let the script download it automatically.
3. Run the script using:

```bash
python image_stitching.py
```

4. The output panoramas will be saved in the `processed_panorama` folder.

## Output

For each object in the dataset, the script creates a panorama using 15 images taken from different angles. The panoramas are saved as `.jpg` files in the `processed_panorama` directory.

## Customization

- **Image Selection**: Modify the `selected_images` logic to select specific images based on angles or other criteria.
- **Parameters**: Adjust parameters for SIFT, BFMatcher, or `cv2.findHomography` to fine-tune the stitching process.

## Issues

- Ensure the dataset folder structure matches the script requirements.
- Homography calculations may fail if there are insufficient feature matches.

## References

- [COIL-20 Dataset](https://www.kaggle.com/datasets/cyx6666/coil20)
- [OpenCV Documentation](https://docs.opencv.org/)

---

**Author:** Shafqat Mehmood
