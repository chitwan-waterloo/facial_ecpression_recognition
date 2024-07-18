import cv2
from keras.models import model_from_json
import numpy as np
# from keras_preprocessing.image import load_img

json_file = open("emotiondetector.json", 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# hear_file = cv2.data.haarcascades + 'haarcascades_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Check if the cascade file was loaded correctly
if face_cascade.empty():
    raise IOError('Cannot load the specified xml file.')

def extract_feature_single(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
labels = {0 : 'angry', 
          1 : 'disgust', 
          2 : 'fear', 
          3 : 'happy', 
          4 : 'neutral', 
          5 : 'sad', 
          6 : 'surprise'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im,1.1,5)
    try:
        for (x,y,w,h) in faces:
            image = gray[y:y+h,x:x+w]
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            image = cv2.resize(image, (48,48))
            img = extract_feature_single(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            print("Predicted output:", prediction_label)

            cv2.putText(im, '% s' %(prediction_label), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))

        cv2.imshow("Output", im)
        cv2.waitKey(27)
    except cv2.error:
        pass