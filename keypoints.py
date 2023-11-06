import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Color conversion
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read() #read the feed
    cv2.imshow('Opencv Feed', frame) #show to screen

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()

