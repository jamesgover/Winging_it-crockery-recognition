import numpy as np
import cv2
import os
FOLDER_NAME = 'frames'


def create_dir(folder=FOLDER_NAME):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print('Error: Creating directory of data')


def get_rate():
    return 100



frame_no = 0
rate = get_rate()
#create_dir()
cap = cv2.VideoCapture(0)
while frame_no < 1000:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame.', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frame_no % rate is 0:
        frame_name = "./" + FOLDER_NAME + "/pot_frame" + str(frame_no) + ".jpg"
        cv2.imwrite(frame_name, frame)
    frame_no += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

