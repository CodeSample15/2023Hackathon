#This file will hold a seperate method used to process data for the main data mining script.
#It will take in a video frame, and return a new frame containing just the mouth.
#Will also take in dimensions for the expected output
import cv2
import mediapipe as mp
import numpy as np

import keyboard
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "../../MPModel/face_landmarker.task"

processed = np.zeros((1, 1, 1))
in_image_dims = [0,0]
out_image_dims = [0,0]

testPos = 200
testing = True

def testthread():
    global testPos
    while testing:
        if keyboard.read_key() == 'n':
            testPos += 1
        if keyboard.read_key() == 'p':
            testPos -= 1
        time.sleep(0.1)

def callback(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #take the landmarks from the result, crop and resize the image
    global processed

    try:
        #211, 450
        if len(result.face_landmarks) != 0:
            x1 = int(result.face_landmarks[0][206].x * in_image_dims[1])
            y1 = int(result.face_landmarks[0][206].y * in_image_dims[0])

            x2 = int(result.face_landmarks[0][369].x * in_image_dims[1])
            y2 = int(result.face_landmarks[0][369].y * in_image_dims[0])

            if x1 > x2:
                t = x1
                x1 = x2
                x2 = t

            if y1 > y2:
                t = y1
                y1 = y2
                y2 = t

            temp = output_image.numpy_view().copy()

            #code for testing different face marks
            #temp = cv2.circle(temp, (x1, y1), 3, (255, 0, 0), -1)
            #cv2.putText(temp, str(testPos), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #processed = temp

            temp = temp[y1:y2, x1:x2]
            temp = cv2.resize(temp, (out_image_dims[1], out_image_dims[0]), interpolation=cv2.INTER_AREA)
            processed = temp

            
    except Exception as error:
        print(error)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=callback)

def segmentMouth(frame, img_width=128, img_height=128):
    global in_image_dims, out_image_dims

    in_image_dims = [frame.shape[0], frame.shape[1]]
    out_image_dims = [img_width, img_height]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    with FaceLandmarker.create_from_options(options) as landmarker:
        landmarker.detect_async(mp_image, 0)