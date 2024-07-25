# Emotion Detection Project

This repository contains the code and resources for an Emotion Detection project. The project aims to detect and classify emotions from facial expressions in real-time using a pre-trained model and OpenCV for face detection.

## Project Structure

- `LICENSE.txt`: License file for the project.
- `emotiondetector.h5`: Trained model file with weights.
- `emotiondetector.json`: Model architecture in JSON format.
- `haarcascade_frontalface_default.xml`: OpenCV classifier for detecting faces in live video.
- `live_detection.py`: Script to use the trained model to detect emotions in real-time via webcam.
- `requirements.txt`: List of dependencies required for the project.
- `train_model.ipynb`: Jupyter notebook used for data cleaning, training using convolution NNs, and saving the model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/chitwan-waterloo/facial_expression_recognition.git
    cd facial_expression_recognition
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, open the `train_model.ipynb` notebook and follow the instructions. This notebook includes data cleaning, preprocessing, model training, and saving the trained model. 

### Real-time Emotion Detection

To run the real-time emotion detection:
```bash
python live_detection.py
```

This script will access your webcam, detect faces, and classify emotions in real-time using a pre-trained model.

## Requirements

The `requirements.txt` file contains the list of dependencies required to run the project. Some key dependencies include:

- TensorFlow
- Keras
- OpenCV
- NumPy

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the `LICENSE.txt` file for details.

## Acknowledgements

This project uses the Haar Cascade Classifier from OpenCV for face detection.