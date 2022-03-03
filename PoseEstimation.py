
import cv2
import mediapipe as mp
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture(0)

prevTime = 0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.9) as pose:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    cv2.imshow('BlazePose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
