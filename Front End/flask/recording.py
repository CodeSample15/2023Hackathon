#Creates a virtual camera that OBS can use to record data
import cv2
from DataProcessing import segmentMouth
import DataProcessing
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 128

# define a video capture object
vid = cv2.VideoCapture(0)

print("Starting virtual camera...")
with pyvirtualcam.Camera(width=IMG_WIDTH, height=IMG_HEIGHT, fps=30) as cam:
    while(True):
        ret, frame = vid.read()
    
        if ret:
            #cv2.imshow('normal', frame)

            # Display the resulting frame
            segmentMouth(frame, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

            processed_frame = DataProcessing.processed
            if not np.array_equal(processed_frame, np.zeros((1, 1, 1))):
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                #cv2.imshow('gray', gray)
                gray = np.stack((gray,)*3, axis=-1)

                cam.send(gray)
            else:
                cam.send(np.zeros((IMG_WIDTH, IMG_HEIGHT, 3)))
            cam.sleep_until_next_frame()

vid.release()
cv2.destroyAllWindows()