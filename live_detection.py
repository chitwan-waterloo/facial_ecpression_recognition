import cv2
from keras.models import model_from_json
import numpy as np
# from keras_preprocessing.image import load_img

# Load the model architecture from JSON file
json_file = open("emotiondetector.json", 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the model weights
model.load_weights("emotiondetector.h5")

# Load the face detection cascade file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Check if the cascade file was loaded correctly
if face_cascade.empty():
    raise IOError('Cannot load the specified XML file.')

def extract_feature_single(image):
    """
    Preprocess the image to be suitable for the model input.
    
    Parameters:
    image (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: The preprocessed image.
    """
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Open a connection to the webcam
webcam = cv2.VideoCapture(0)

# Emotion labels corresponding to the model output
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Main loop to continuously get frames from the webcam
while True:
    # Read a frame from the webcam
    i, im = webcam.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(im, 1.1, 5)
    
    try:
        for (x, y, w, h) in faces:
            # Extract the region of interest (the face) from the grayscale image
            image = gray[y:y+h, x:x+w]
            
            # Draw a rectangle around the detected face
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Resize the face image to 48x48 pixels, which is the input size for the model
            image = cv2.resize(image, (48, 48))
            
            # Preprocess the image
            img = extract_feature_single(image)
            
            # Predict the emotion using the model
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Print the predicted emotion
            print("Predicted output:", prediction_label)
            
            # Put the predicted emotion label on the image
            cv2.putText(im, '% s' % (prediction_label), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        # Display the resulting frame
        cv2.imshow("Output", im)
        
        # Wait for 27 ms before moving on to the next frame
        cv2.waitKey(27)
    except cv2.error:
        pass
