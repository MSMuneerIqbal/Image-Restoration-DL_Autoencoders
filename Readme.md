# Impulse Noise Detection and Denoising using CNNs

This repository contains an end-to-end project for restoring images corrupted with impulse (salt-and-pepper) noise. The workflow includes:

-   **Data Loading and Preprocessing:** Load clean RGB images, convert them to grayscale, and normalize them.
-   **Impulse Noise Addition:** Apply salt-and-pepper noise to the images and generate corresponding binary noise masks.
-   **Noise Detection:** Train a CNN segmentation model (U-Net–like architecture) to predict the impulse noise mask.
-   **Denoising:** Train a CNN autoencoder to remove the noise and restore the clean image using Mean Squared Error (MSE) loss.
-   **Classical Filtering Comparison:** Apply median and fuzzy filters as baseline methods for denoising.
-   **Evaluation Metrics:** Evaluate restoration performance using PSNR and SSIM metrics.
-   **Visualization & Saving:** Plot training curves (loss and accuracy) and save the resulting noisy and denoised images to directories.

## Dataset

You can use your own dataset by placing your images in a folder named `BSD500` in the repository's root. Alternatively, you can try a publicly available dataset on Kaggle. For example, you might consider using the [BSD500 Dataset](INSERT_KAGGLE_LINK_HERE) as a starting point.

## Features

-   **Impulse Noise Generation:** Adds salt-and-pepper noise to clean images and produces ground-truth masks.
-   **Noise Detection Model:** A U-Net–like CNN segmentation model that learns to predict the noise mask.
-   **Denoising Autoencoder:** A CNN-based autoencoder that learns to restore the clean image from the noisy input using MSE loss.
-   **Classical Methods:** Comparison with median filtering and a custom fuzzy filtering algorithm.
-   **Performance Evaluation:** PSNR and SSIM metrics are computed to quantitatively evaluate the restoration quality.
-   **Training Visualizations:** Training graphs for both the noise detection model (loss & accuracy) and the denoising autoencoder (loss & MSE) are plotted.
-   **Saving Outputs:** Noisy images are saved in the `noisy` folder and denoised images in the `denoised` folder.

## Requirements

-   Python 3.x
-   TensorFlow 2.x
-   NumPy
-   OpenCV (cv2)
-   Matplotlib
-   scikit-learn

You can install the required packages using:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```
## How to Run

1.  **Place Your Dataset:** Put your clean images in a folder named `BSD500` located in the project’s root directory.
2.  **Run the Notebook/Script:** Execute the provided code (either in a Jupyter Notebook or as a Python script) to:
    -   Load and preprocess the images.
    -   Add impulse noise and generate noise masks.
    -   Split the data into training and testing sets.
    -   Train the noise detection and denoising models.
    -   Evaluate performance with PSNR and SSIM.
    -   Save the noisy and denoised images to their respective folders.
3.  **Review Training Graphs:** Training curves for both models (loss & accuracy/MSE) are plotted for analysis.

## Code Explanation

**Step 1: Import Libraries and Set Up**

The necessary libraries are imported, and global parameters (image size, noise probabilities, training hyperparameters) are defined.

**Step 2: Load and Preprocess the Dataset**

Images are loaded from the `BSD500` folder, decoded using `tf.image.decode_jpeg` for known shape, resized to 128×128, converted to grayscale, and normalized.

**Step 3: Add Salt-and-Pepper Noise**

Impulse noise is added by randomly setting a fraction of pixels to 1 (salt) or 0 (pepper). A binary mask is generated to mark noisy pixels. Noisy images are then saved to the `noisy` directory.

**Step 4: Visualization**

A sample clean image, its noisy version, and the corresponding ground-truth noise mask are displayed.

**Step 5: Split Data**

The dataset is split into training and testing sets (80/20 split).

**Step 6: Noise Detection Model**

A CNN segmentation model (similar to a U-Net) is defined and compiled with binary crossentropy loss to predict the noise mask.

**Step 7: Train Noise Detection Model**

The model is trained on the noisy images and noise masks. Training curves (loss and accuracy) are plotted and sample predictions are visualized.

**Step 8: Denoising Autoencoder**

A CNN autoencoder is defined to restore the clean image from the noisy input using MSE loss. Training curves are plotted.

**Step 9: Evaluation**

PSNR and SSIM metrics are computed for the denoised images to evaluate restoration quality.

**Step 10: Classical Filtering and Visual Comparison**

Median and fuzzy filters are applied as classical denoising methods. Their performance is compared using PSNR, and visual results (including error maps) are displayed. Finally, the denoised images are saved to the `denoised` folder.

