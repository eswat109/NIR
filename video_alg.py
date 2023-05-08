import numpy as np
import cv2 as cv
from enum import Enum

from algo import *

class MaskedVideoGenerator:

    class WorkMethod(Enum):
        Default = 0
        Detail = 1

    MASK_ONLY = True
    frame_gap = 1
    method = WorkMethod.Detail

    def concatFrame(self, f, SPEC, ED, AD):
        img_1 = cv.hconcat([f, SPEC])
        img_2 = cv.hconcat([ED, AD])
        res = cv.vconcat([img_1, img_2])
        return res

    def getFrameToWrite(self, slvr, frame, mask, ED = None, AD = None):
        res = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        if not self.MASK_ONLY:
            if self.method == self.WorkMethod.Detail:
                res = self.concatFrame(frame,
                                 slvr.combineFrameAndMask(frame, mask),
                                 slvr.combineFrameAndMask(frame, ED),
                                 slvr.combineFrameAndMask(frame, AD))
            elif self.method == self.WorkMethod.Default:
                res = slvr.combineFrameAndMask(frame, mask)
        else:
            if self.method == self.WorkMethod.Detail:
                res = self.concatFrame(frame,
                                       cv.cvtColor(mask, cv.COLOR_GRAY2RGB),
                                       cv.cvtColor(ED, cv.COLOR_GRAY2RGB),
                                       cv.cvtColor(AD, cv.COLOR_GRAY2RGB))
        return res


    def main(self):
        vid_dir = 'vid/'

        vid_name = 'video_' + str(15)
        vid_format = '.mp4'

        vid_name = 'Cabinet2_F'
        vid_name = 'Cabinet2_F_Sq'
        vid_name = 'Cabinet2_F_Sp'
        vid_name = 'Cabinet3_F_Sp'
        vid_format = '.avi'

        #slvr = Solver(0.03, 10, 0.7, 0.7, 30, 20)
        #slvr = Solver(0.03, 10, 0.2, 0.7, 0, 25)
        kwargs = {'SIFT_peak': 0.09,
                  'SIFT_edge': 8,
                  'SIFT_feature_elimination': 20,
                  'Samson_err': 0.02,
                  'Ratio': 0.7,
                  }
        slvr = Solver(**kwargs)

        cap = cv.VideoCapture(vid_dir + vid_name + vid_format)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        frame_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        if self.method == self.WorkMethod.Detail:
            frame_w *= 2
            frame_h *= 2

        new_name = vid_dir + 'masked/' + vid_name + '_masked_' + str(self.frame_gap) + '__' + \
                   str(slvr.SIFT_peak_threshold) + '_' + str(slvr.SIFT_edge_threshold) + '_' + \
                   str(slvr.samson_err) + '_' + str(slvr.ratio_test) + '__' + str(slvr.Gaussian_deviation) + '_' + str(slvr.point_size) + '.avi'
        # out = cv.VideoWriter(new_name, fourcc, fps, (frame_w, frame_h))
        out = cv.VideoWriter(new_name, fourcc, fps, (int(frame_w), int(frame_h)))
        frame = prev_frame = None
        mask_SPEC, mask_ED, mask_AD = None, None, None
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
            if frame_i < self.frame_gap:
                if mask_SPEC is None:
                    out.write(frame)
                else:
                    masked_frame = self.getFrameToWrite(slvr, frame, mask_SPEC, mask_ED, mask_AD)
                    out.write(masked_frame)
                continue
            frame_i = -1

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray_prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

            try:
                mask_SPEC, mask_ED, mask_AD = slvr.getMasksFromFrames(gray_prev_frame, gray_frame)
            except:
                mask_SPEC, mask_ED, mask_AD = None, None, None
                out.write(frame)
                prev_frame = frame
                continue

            masked_frame = self.getFrameToWrite(slvr, frame, mask_SPEC, mask_ED, mask_AD)
            out.write(masked_frame)

            prev_frame = frame.copy()
        cap.release()

if __name__ == '__main__':
    gen = MaskedVideoGenerator()
    gen.main()