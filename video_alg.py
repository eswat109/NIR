import numpy as np
import cv2 as cv

from algo import *

MASK_ONLY = False

def getFrameToWrite(frame, mask, slvr):
    res = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    if not MASK_ONLY:
        res = slvr.combineFrameAndMask(frame, mask)
    return res

def main():
    frame_gap = 1
    vid_dir = 'vid/'

    vid_name = 'video_' + str(8)
    vid_format = '.mp4'
    vid_name = 'Cabinet2_Full'
    vid_format = '.avi'

    slvr = Solver()

    cap = cv.VideoCapture(vid_dir + vid_name + vid_format)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    new_name = vid_dir + 'masked/' + vid_name + '_masked_' + str(frame_gap) + '__' + \
               str(slvr.SIFT_peak_threshold) + '_' + str(slvr.SIFT_edge_threshold) + '_' + \
               str(slvr.samson_err) + '_' + str(slvr.ratio_test) + '__' + str(slvr.Gaussian_deviation) + '_' + str(slvr.point_size) + '.avi'
    # out = cv.VideoWriter(new_name, fourcc, fps, (frame_w, frame_h))
    out = cv.VideoWriter(new_name, fourcc, fps, (int(frame_w), int(frame_h)))
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
                masked_frame = getFrameToWrite(frame, mask, slvr)
                out.write(masked_frame)
            continue
        frame_i = -1

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

        '''
        mask, _1, _2 = slvr.getMasksFromFrames(gray_prev_frame, gray_frame)
        '''
        try:
            mask, _1, _2 = slvr.getMasksFromFrames(gray_prev_frame, gray_frame)
        except:
            mask = None
            out.write(frame)
            prev_frame = frame
            continue

        masked_frame = getFrameToWrite(frame, mask, slvr)
        out.write(masked_frame)

        prev_frame = frame
    cap.release()

if __name__ == '__main__':
    main()