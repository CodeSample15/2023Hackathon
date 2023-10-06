#just some testing driver code for now
import cv2
from DataProcessing import segmentMouth
import DataProcessing
import os
import sys
import numpy as np
import time
import pyaudio
import wave
import threading

filename = input("Enter output file name: ")
if os.path.isfile(filename):
    res = input("WARNING! FILE ALREADY EXISTS WITH THIS NAME. PRESS ENTER TO PROCEED")
    if res == 'stop':
        sys.exit()

#for video recording
fps = 20
frameSize = (128,128)
video_filename = "temp_video.mp4"
video_writer = cv2.VideoWriter_fourcc(*'DIVX')
video_out = cv2.VideoWriter(video_filename, video_writer, fps, frameSize)

#for recording sounds
isopen = True
rate = 44100
frames_per_buffer = 1024
channels = 1
pformat = pyaudio.paInt16
audio_filename = "temp_audio.wav"
audio = pyaudio.PyAudio()
stream = audio.open(format=pformat,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer = frames_per_buffer)
audio_frames = []


def audioRecordThread():
    global audio_frames
    stream.start_stream()
    while(isopen == True):
        data = stream.read(frames_per_buffer) 
        audio_frames.append(data)
        if not isopen:
            break

# define a video capture object
vid = cv2.VideoCapture(0)

frame_count = 0
start_time = time.time()

record_audio = threading.Thread(target=audioRecordThread)
record_audio.start()

while(True):
    ret, frame = vid.read()
  
    cv2.imshow('normal', frame)

    # Display the resulting frame
    segmentMouth(frame)

    processed_frame = DataProcessing.processed
    if not np.array_equal(processed_frame, np.zeros((1, 1, 1))):
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('gray', gray)
        video_out.write(gray)
        frame_count += 1

        #cv2.imshow('frame', DataProcessing.processed)
      
    if cv2.waitKey(1) & 0xFF == ord('q'): #33.3 -> 30 frames per second
        break

vid.release()
video_out.release()
cv2.destroyAllWindows()

#save and export data
isopen = False
record_audio.join()

stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(audio_filename, 'wb')
waveFile.setnchannels(channels)
waveFile.setsampwidth(audio.get_sample_size(pformat))
waveFile.setframerate(rate)
waveFile.writeframes(b''.join(audio_frames))
waveFile.close()

elapsed_time = time.time() - start_time
recorded_fps = frame_count / elapsed_time

'''
if abs(recorded_fps - 30) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
    cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.mp4"
    os.system(cmd)

    cmd = "ffmpeg -ac 1 -channel_layout stereo -i temp_audio.wav -i temp_video2.mp4 -pix_fmt yuv420p " + filename + ".avi"
    os.system(cmd)
else:
    cmd = "ffmpeg -ac 1 -channel_layout stereo -i temp_audio.wav -i temp_video.mp4 -pix_fmt yuv420p " + filename + ".avi"
    os.system(cmd)
'''
#cleanup
#os.remove("temp_audio.wav")
#os.remove("temp_video.mp4")
#os.remove("temp_video2.mp4")