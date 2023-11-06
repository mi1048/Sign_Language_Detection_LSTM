import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Color conversion BGR to RGB
    image.flags.writeable = False                    
    results = model.process(image)                 #Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Color conversion
    return image, results

def draw_Landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.HAND_CONNECTIONS)
    #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,155,10), thickness = 1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(155,80,10), thickness = 4, circle_radius=4)
                              )    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,155,10), thickness = 1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(155,80,10), thickness = 4, circle_radius=4)
                              )    



cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read() #read the feed

        image, results = mediapipe_detection(frame, holistic)
        
        draw_styled_landmarks(image, results)

        cv2.imshow('Opencv Feed', image) #show to screen

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release() 
    cv2.destroyAllWindows()

draw_Landmarks(frame, results)

#plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten if results.right_hand_landmarks else np.zeros(21*3)

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


DATA_PATH = os.path.join('MP_Data') # path for exported data, numpy arrays
actions = np.array(['hello', 'thanks', 'iloveyou']) # actions to detect
no_sequences = 40 # numbers of videos in data
sequence_lenght = 40 # frames in lenght video

