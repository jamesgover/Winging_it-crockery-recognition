import numpy as np
import cv2
import os
import pandas as pd
import masks_obj_id as masks
from skimage import io as skio

FRAMES_FOLDER = 'frames'  # where to save photos by default
DATA_FILE = 'sample_photos_data.csv'  # the file that stores data
PHOTO_FOLDER = 'sample_photos'  # directory where program will find photos to use as test set





''' a wrapper for os.makedirs to handle errors
@:param folder, the folder to try to make
'''
def create_dir(folder=FRAMES_FOLDER):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print('Error: Creating directory of data')


''' function to save data, allows for scanning, redirecting, duplicating, printing and ect. to be implemented
@:param data, the data to be saved as a csv'''
def save_data(data):
    data.to_csv(path_or_buf=DATA_FILE)


'''
@:param file, the file to be loaded
@:return a panda.Dataframe instance capturing the loaded data
'''
def load_data(file):
    data = masks.create_pd_frame(original=False)
    try:
        loaded_data = pd.read_csv(filepath_or_buffer=file)
        data = data.append(loaded_data, ignore_index=True, sort=True)
    except IOError:
        print("Error: error loading data.")
    return data


''' function to be implemented when adapting for a live stream application (attatched to washing machine) 
instead of just a batch system as we tested it as
@returns the rate (in frames/(screen_size - crockery_length))'''
def get_rate():
    return 100


''' uses the system camera (webcam for a laptop) to take a video and process frames and stores the 
resultant data in DATA_FILE to be fed into recognition.py during operation under stream conditions
this is NOT used for batch processing
'''
def capture_and_process():
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


''' iterated over photos in DATA_FILE and (currently) sets creates a labeled data set by querying the user
'''
def train(data_file=DATA_FILE, photo_folder=PHOTO_FOLDER):
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


# process_sample_photos()

