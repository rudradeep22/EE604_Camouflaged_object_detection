# Camouflaged Object Detection

This project aims to detect camouflaged objects in images using a deep learning model. The model used is BiRefNet, a pre-trained image segmentation model from the `transformers` library.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/camouflaged-object-detection.git
    cd camouflaged-object-detection
    ```

2. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Command Line Interface

1. Run the `main.py` script to predict the mask for an image:
    ```sh
    python main.py
    ```

2. Follow the prompts to enter the device (cpu/cuda) and view the predicted mask and IoU score (if ground truth is provided).

### Web Interface

1. Run the `app.py` script to launch the Gradio web interface:
    ```sh
    python app.py
    ```

2. Open the provided URL in your browser to upload an image and optionally a ground truth mask to generate predictions and calculate the IoU score.

## Project Structure

- `app.py`: Contains the Gradio web interface for uploading images and generating predictions.
- `main.py`: Command line interface for predicting masks and calculating IoU scores.
- `requirements.txt`: List of required Python packages.
- `data/`: Directory to store input images, ground truth masks, and output predictions.

## Model Details

The project uses the BiRefNet model for image segmentation, which is loaded from the `transformers` library. The model is fine-tuned for detecting camouflaged objects in images.
