import numpy as np
import cv2 as cv
from enum import Enum

from MaskExtractor import *

class VideoHandler:

    class WorkMethod(Enum):
        Default = 0
        Detail = 1

    MASK_ONLY = False
    frame_gap = 0
    method = WorkMethod.Default

    kwargs = {'SIFT_peak': 0.03,
              'SIFT_edge': 7,
              'SIFT_feature_elimination': 15,
              'Samson_err': 2.5,
              'Trees': 5,
              'P_size': 20,
              }
    S_err_2 = 2.5

    def concatFrame(self, f, SPEC, ED, AD):
        img_1 = cv.hconcat([f, SPEC])
        img_2 = cv.hconcat([ED, AD])
        res = cv.vconcat([img_1, img_2])
        return res

    def culculation(self, frame1, frame2):
        kwargs = self.kwargs
        slvr = MaskExtractor(**kwargs)
        slvr.F_intens_on = True
        slvr.F_max_only = True
        slvr.F_andMethod = slvr.AndMethod.MIN

        kwargs['Samson_err'] = self.S_err_2
        slvr_2 = MaskExtractor(**kwargs)
        slvr_2.F_intens_on = True
        slvr_2.F_max_only = True
        slvr_2.F_andMethod = slvr.AndMethod.MIN

        mask_SPEC, mask_ED, mask_AD = slvr.getMasksFromFrames(frame1, frame2)
        return mask_SPEC, mask_ED, mask_AD

        '''mask_SPEC_2, mask_ED_2, mask_AD_2 = slvr_2.getMasksFromFrames(frame1, frame2)
        mask_SPEC_R = slvr.combineMasks(mask_ED, mask_AD_2)

        return mask_SPEC_R, mask_ED, mask_AD_2'''


    def getFrameToWrite(self, frame, mask, ED, AD):
        slvr = MaskExtractor()
        res = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        if not self.MASK_ONLY:
            if self.method == self.WorkMethod.Detail:
                '''res = self.concatFrame(frame,
                                 slvr.combineFrameAndMask(frame, mask),
                                       cv.cvtColor(ED, cv.COLOR_GRAY2RGB),
                                       cv.cvtColor(AD, cv.COLOR_GRAY2RGB))'''
                res = self.concatFrame(frame,
                                       slvr.combineFrameAndMask(frame, mask),
                                       slvr.combineFrameAndMask(frame, ED),
                                       slvr.combineFrameAndMask(frame, AD)
                                       )
                '''res = self.concatFrame(slvr.combineFrameAndMask(frame, slvr.getMaxContourMask(mask)),
                                       slvr.combineFrameAndMask(frame, mask),
                                       cv.cvtColor(ED, cv.COLOR_GRAY2RGB),
                                       cv.cvtColor(AD, cv.COLOR_GRAY2RGB))'''
            elif self.method == self.WorkMethod.Default:
                res = slvr.combineFrameAndMask(frame, mask)
                #res = slvr.combineFrameAndMask(frame, slvr.getMaxContourMask(mask))
        else:
            if self.method == self.WorkMethod.Detail:
                res = self.concatFrame(frame,
                                       cv.cvtColor(mask, cv.COLOR_GRAY2RGB),
                                       cv.cvtColor(ED, cv.COLOR_GRAY2RGB),
                                       cv.cvtColor(AD, cv.COLOR_GRAY2RGB))
        return res


    def selectSpecularArea(self, inputVideoName):

        self.MASK_ONLY = False
        self.method = self.WorkMethod.Default

        cap = cv.VideoCapture(inputVideoName)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        frame_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        if self.method == self.WorkMethod.Detail:
            frame_w *= 2
            frame_h *= 2
        '''
        new_name = vid_dir + 'masked/' + vid_name + '_masked_' + str(self.frame_gap) + '__' + \
                    str(self.kwargs['SIFT_peak']) + '_' + str(self.kwargs['SIFT_edge']) + '_' + \
                    str(self.kwargs['SIFT_feature_elimination']) + '_' + str(self.kwargs['Samson_err']) + '_' + str(self.S_err_2) + '_' + \
                    '.avi'

        '''
        # str(slvr.AD_MULT) + '_' + str(slvr.point_size) + '_' + str(slvr.Specular_thr) + \
        # '__' +  \

        new_name = 'Result.avi'

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
                    masked_frame = self.getFrameToWrite(frame, mask_SPEC, mask_ED, mask_AD)
                    out.write(masked_frame)
                continue
            frame_i = -1

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray_prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

            try:
                mask_SPEC, mask_ED, mask_AD = self.culculation(gray_prev_frame, gray_frame)
            except:
                mask_SPEC, mask_ED, mask_AD = None, None, None
                out.write(frame)
                prev_frame = frame
                continue

            masked_frame = self.getFrameToWrite(frame, mask_SPEC, mask_ED, mask_AD)
            out.write(masked_frame)

            prev_frame = frame.copy()
        cap.release()

if __name__ == '__main__':
    gen = VideoHandler()

    vid_dir = 'vid/'

    vid_name = 'video_' + str(10)
    vid_format = '.mp4'

    vid_name = 'Cabinet2_F_Sp'
    vid_name = 'Cabinet3_F_Sp'
    vid_name = 'Cabinet3_AF1_F'
    vid_name = 'Cabinet2_F_Sq'
    vid_name = 'Cabinet2_Sq_Short_F'
    vid_name = 'Cabinet2_Full'
    vid_name = 'Cabinet2_Sq_Short_2_F'
    vid_format = '.avi'

    vidInput = vid_dir + vid_name + vid_format
    gen.selectSpecularArea(vidInput)