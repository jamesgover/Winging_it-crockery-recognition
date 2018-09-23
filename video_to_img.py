import numpy as np
import cv2
import os
import pandas as pd
#from py_computer_vision
import masks_obj_id as masks

FOLDER_NAME = 'frames'
DATA_FILE = 'file_data.csv'


def create_dir(folder=FOLDER_NAME):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print('Error: Creating directory of data')


def save_data(data):
    data.to_csv(path_or_buf=DATA_FILE)


def load_data(file):
    data = masks.create_pd_frame(original=False)
    try:
        loaded_data = pd.read_csv(filepath_or_buffer=file)
        data = data.append(loaded_data, ignore_index=True, sort=True)
    except IOError:
        print("Error: error loading data.")
    return data



def get_rate():
    return 100


def main():
    frame_no = 0
    rate = get_rate()
    # create_dir()
    cap = cv2.VideoCapture(0)
    data = load_data(DATA_FILE)
    while frame_no < 1000:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Display the resulting frame
        cv2.imshow('frame.', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_no % rate is 0:
            #   THIS SHOULD BE THREADED
            # save image
            frame_name = "./" + FOLDER_NAME + "/pot_frame" + str(frame_no) + ".jpg"
            cv2.imwrite(frame_name, frame)
            new_data = masks.img_to_data(frame)
            masks.plot(masks.sep_and_strip_img(masks.get_mask(frame)))
            data = data.append(new_data, ignore_index=True, sort=True)

            # process image
        frame_no += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # save data as csv
    save_data(data)


# main()

