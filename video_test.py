import numpy as np
import cv2 as cv

from algo import *

if __name__ == '__main__':
    frame_gap = 1
    vid_dir = 'vid_test/'
    vid_name_masked = 'Cabinet2_Full'
    vid_name_true = 'Cabinet2_Full'
    vid_format = '.avi'

    cap = cv.VideoCapture(vid_dir + vid_name_masked + vid_format)

    frame = prev_frame = None
    mask = None
    frame_i = -1
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_i += 1
        if prev_frame is None:
            prev_frame = frame
            continue
        if frame_i < frame_gap:
            if mask is not None:
                masked_frame = combineFrameAndMask(frame, mask)
            continue
        frame_i = -1

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        (mask, _) = getMasksFromFrames(gray_prev_frame, gray_frame)
        masked_frame = combineFrameAndMask(frame, mask)

        prev_frame = frame
    cap.release()
    #cv.destroyAllWindows()