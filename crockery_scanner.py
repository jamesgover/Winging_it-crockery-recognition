"""
Created on Mon Sep 10 00:36:59 2018
@author: hokl3
"""

import cv2
import time
import masks_obj_id as masks
import video_to_img as v2i


class CrockeryScanner:

    def __init__(self, width=640, height=480):
        self.dimensions = (width, height)
        return

    def get_fps(self, video, num_frames=60):
        # num_frames is Number of frames to capture

        # Start time
        start = time.time()

        # Grab a few frames
        for i in range(0, num_frames):
            ret, frame = video.read()

        # End time
        end = time.time()

        # Time elapsed
        seconds = end - start

        # Calculate frames per second
        fps = num_frames / seconds
        self.fps = fps
        return fps

    def scan_image(self, time_threshold=1 / 3, num_frames=60, video="webcam"):
        masks.vid_area()
        # get the webcam:
        if video is "webcam)":
            cap = cv2.VideoCapture(0)
            cap.set(3, self.dimensions[0])
            cap.set(4, self.dimensions[1])
            time.sleep(2)
        else:
            print("using video: "+ video)
            cap = cv2.VideoCapture(video)

        fps = self.get_fps(cap, num_frames)

        def decode(image):
            """
            :param image:
            :return:list(Region) region_objects a list of named tuple containing the location
                    and data of the regions contained within image
            """
            # Find barcodes and QR codes
            r_objs = masks.decode(image)
            # Print results
            for region in r_objs:
                silent = True
                if not silent:
                    print('region data : ', region.data)
                    print('position ', region.location)
            return r_objs

        result = []
        result_dataframe = masks.create_pd_frame()
        y_boundary = 50 # top of the frame
        empty_frame = 0
        success = True

        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            cv2.imwrite("./data/frame.jpg", frame)
            if ret:
                cv2.imshow('frame', frame)
                # Operations on frame
                # im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                region_objects = decode(frame)
                masks.plot(masks.get_mask(frame))
                # print(str(region_objects))
                cv2.imshow('Frame', frame)
                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break


            if empty_frame >= fps * time_threshold and region_objects == []:
                empty_frame += 1
                possible_new_boundaries = []  # set boundary to top of the frame
            elif region_objects == []:
                empty_frame += 1
                possible_new_boundaries = [y_boundary]
            else:
                possible_new_boundaries = [y_boundary]

            # Define new namedtuple to create new decodedObject

            region_objects = sorted(region_objects, key=lambda obj: obj.trailing_edge, reverse=True)
            for region in region_objects:

                empty_frame = 0

                if region.trailing_edge < y_boundary:
                    # Only record objects codes that are below of previous topmost Object
                    print('region data : ', region.data)
                    print('trailing edge : ', region.trailing_edge)
                    print('position ', region.location)
                    print("---------------------")
                    result.append(region.data)
                    result_dataframe.append(region.data, ignore_index=True, sort=True)
                possible_new_boundaries.append(region.trailing_edge)

            y_boundary = min(possible_new_boundaries)
            # Display the resulting frame
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):  # wait for 's' key to save
                cv2.imwrite('Capture.png', frame)

                # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        v2i.save_data(result_dataframe, "data/live_results")
        print(result)

if __name__ == "__main__":
    cs = CrockeryScanner()
    cs.scan_image(video="./data/sampleVid.mp4")

