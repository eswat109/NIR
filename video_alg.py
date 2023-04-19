import numpy as np
import cv2 as cv

from algo import *

if __name__ == '__main__':
    frame_gap = 1
    vid_dir = 'vid/'
    vid_name = 'video_' + str(10)
    vid_format = '.mp4'
    cap = cv.VideoCapture(vid_dir + vid_name + vid_format)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    new_name = vid_dir + 'masked/' + vid_name + '_masked_' + str(frame_gap) + '__' + \
        str(SIFT_peak_threshold) + '_' + str(SIFT_edge_threshold) + '_' + \
        str(samson_err) + '_' + str(ratio_test) + '__' + str(Gaussian_deviation) + '_' + str(point_size) + '.avi'
    #out = cv.VideoWriter(new_name, fourcc, fps, (frame_w, frame_h))
    out = cv.VideoWriter(new_name, fourcc, fps, (int(frame_w),  int(frame_h)))
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
            if mask is None:
                out.write(frame)
            else:
                frame_mask = combineFrameAndMask(frame, mask)
                out.write(frame_mask)
            continue
        frame_i = -1

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        _1, _2, mask = getMasksFromFrames(gray_prev_frame, gray_frame)
        frame_mask = combineFrameAndMask(frame, mask)

        out.write(frame_mask)

        prev_frame = frame
    cap.release()
    #cv.destroyAllWindows()