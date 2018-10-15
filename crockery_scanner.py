# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 00:36:59 2018
@author: hokl3
"""

import cv2
import time
from collections import namedtuple
import masks_obj_id as masks


class meal_scanner:

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

    def scan_codes(self, time_threshold=1 / 3, num_frames=60):
        # get the webcam:
        cap = cv2.VideoCapture(0)
        cap.set(3, self.dimensions[0])
        cap.set(4, self.dimensions[1])
        time.sleep(2)

        fps = self.get_fps(cap, num_frames)

        def decode(im):
            # Find barcodes and QR codes
            decodedObjects = masks.decode(im)
            # Print results
            for obj in decodedObjects:
                print('Type : ', obj.type)
                print('Data : ', obj.data)
                print('Points : ', obj.rect, '\n')
            return decodedObjects

        result = []
        xboundary = self.dimensions[1]
        empty_frame = 0

        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Operations on frame
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            decodedObjects = decode(im)

            if empty_frame >= fps * time_threshold and decodedObjects == []:
                empty_frame += 1
                possible_new_boundaries = [self.dimensions[1]]
            elif decodedObjects == []:
                empty_frame += 1
                possible_new_boundaries = [xboundary]
            else:
                possible_new_boundaries = [xboundary]

            # Define new namedtuple to create new decodedObject
            DecodedNew = namedtuple('Decoded', ['data', 'type', 'rect', 'polygon', 'xcentre'])

            decodedObjectsNew = []
            for decodedObject in decodedObjects:
                # Add xcentre to decodedObjects
                points = decodedObject.rect

                objdict = decodedObject._asdict()
                objdict['xcentre'] = points.left + points.width / 2

                decodedObjectsNew.append(DecodedNew(**objdict))

            decodedObjectsNew = sorted(decodedObjectsNew, key=lambda obj: obj.xcentre, reverse=True)

            for decodedObject in decodedObjectsNew:
                if decodedObject.type != 'QRCODE':
                    continue
                # x_centre = decodedObject.xcentre
                x_left = decodedObject.rect.left

                empty_frame = 0

                if x_left < xboundary:
                    # Only record QR codes that are to the right of previous leftmost QR code.
                    result.append(str(decodedObject.data)[2:-1])
                possible_new_boundaries.append(x_left)

            xboundary = min(possible_new_boundaries)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):  # wait for 's' key to save
                cv2.imwrite('Capture.png', frame)

                # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        print(result)
        return result


