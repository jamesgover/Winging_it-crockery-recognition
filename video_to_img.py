import numpy as np
import cv2
import os
import pandas as pd
import masks_obj_id as masks
from skimage import io as skio

FRAMES_FOLDER = 'frames'  # where to save photos by default
DATA_FILE = masks.DATA_FILE  # the file that stores data
PHOTO_FOLDER = masks.PHOTO_FOLDER  # directory where program will find photos to use as test set


def create_dir(folder=FRAMES_FOLDER):
    '''
    a wrapper for os.makedirs to handle errors
    :param folder: the folder to try to make
    :return: None
    '''
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print('Error: Creating directory of data')


def save_data(data, file=DATA_FILE):
    '''
    function to save data, allows for scanning, redirecting, duplicating, printing and ect. to be implemented
    :param data: the data to be saved as a csv
    :param file: the file to save the data to
    :return: None
    '''
    data.to_csv(path_or_buf=file)


def load_data(file):
    '''
    load the data that is in file
    :param file: the file to be loaded
    :return: panda.Dataframe instance capturing the loaded data
    '''
    data = masks.create_pd_frame(original=False)
    try:
        loaded_data = pd.read_csv(filepath_or_buffer=file)
        data = data.append(loaded_data, ignore_index=True, sort=True)
    except IOError:
        print("No existing data file found")
    return data


def get_rate():
    '''
    function to be implemented when adapting for a live stream application (attatched to washing machine)
    instead of just a batch system as we tested it as
    :return: the rate (in frames/(screen_size - crockery_length))
    '''
    return 100


def live_train():
    '''
    uses the system camera (webcam for a laptop) to take a video and process frames and stores the
    resultant data in DATA_FILE to be fed into recognition.py during operation under stream conditions
    this is NOT used for batch processing
    :return: nothing
    '''
    masks.set_train(True)
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
            frame_name = "./" + FRAMES_FOLDER + "/pot_frame" + str(frame_no) + ".jpg"
            cv2.imwrite(frame_name, frame)
            new_data = masks.img_to_data(frame)
            # masks.plot(masks.sep_and_strip_img(masks.get_mask(frame)))
            data = data.append(new_data, ignore_index=True, sort=True)

            # process image
        frame_no += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # save data as csv
    save_data(data)


def train(data_file=DATA_FILE, photo_folder=PHOTO_FOLDER):
    '''
    iterated over photos in photo_folder and (currently) sets creates a labeled data set by querying the user
    :param data_file: the file to save the data to
    :param photo_folder: the folder to search for photos
    :return: None
    '''
    masks.set_train(True)
    data = load_data(data_file)
    for filename in os.listdir(photo_folder):
        if filename.endswith(".jpg"):
            image_name = os.path.join(photo_folder, filename)
            image = skio.imread("./" + photo_folder + '/' + filename)
            masks.plot(image)
            new_data = masks.img_to_data(image)
            data = data.append(new_data, ignore_index=True, sort=True)
            continue
        else:
            continue
    # save data as csv
    save_data(data)


if __name__ == "__main__":
    train()

