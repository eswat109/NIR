import numpy as np
import cv2 as cv
import pandas as pd
import xlsxwriter

from algo import *


def my_median(sample):
     n = len(sample)
     index = n // 2
     # Sample with an odd number of observations
     if n % 2:
         return sorted(sample)[index]
     # Sample with an even number of observations
     return sum(sorted(sample)[index - 1:index + 1]) / 2


class MaskedVideoTester:
    frame_gap = 1
    vid_dir = 'vid_test/'

    vid_name_orig = 'video_' + str(8)
    vid_format = '.mp4'
    vid_name_orig = 'Cabinet2_F_Sq'
    vid_name_mask = 'Cabinet2_M_Sq'
    vid_name_orig = 'Cabinet3_F_Sp'
    vid_name_mask = 'Cabinet3_M_Sp'
    vid_format = '.avi'

    def main(self, slvr):
        cap_o = cv.VideoCapture(self.vid_dir + self.vid_name_orig + self.vid_format)
        cap_m = cv.VideoCapture(self.vid_dir + self.vid_name_mask + self.vid_format)
        frame = prev_frame = None
        mask = None
        frame_i = -1
        precision_total = []
        recall_total = []
        while cap_o.isOpened() and cap_m.isOpened():
            ret, frame = cap_o.read()
            # if frame is read correctly ret is True
            if not ret:
                #print("Can't receive frame (stream end?). Exiting ...")
                break

            ret, mask_true = cap_m.read()
            # if frame is read correctly ret is True
            if not ret:
                #print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_i += 1
            if prev_frame is None:
                prev_frame = frame
                continue
            if frame_i < self.frame_gap:
                continue
            frame_i = -1

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray_prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

            try:
                mask, _1, _2 = slvr.getMasksFromFrames(gray_prev_frame, gray_frame)
            except:
                mask = None
                prev_frame = frame
                continue

            mask_true = cv.cvtColor(mask_true, cv.COLOR_BGR2GRAY)
            mask_test = mask
            precision, recall = slvr.getPrecisionRecall(mask_test, mask_true)
            precision_total.append(precision)
            recall_total.append(recall)

            prev_frame = frame
        cap_o.release()
        cap_m.release()
        return (precision_total, recall_total)

class ParameterRange:

    def __init__(self, from_, to_, step_):
        self.from_ = from_
        self.to_ = to_
        self.step_ = step_

def printTitle(worksheet):
    var = ['SIFT_peak', 'SIFT_edge', 'SIFT_feature', 'Sampson_err', 'Ratio', 'Avg Precision', 'Avg Recall',
           'M Precision', 'M Recall']
    for i, v in enumerate(var):
        worksheet.write(0, i, v)

def printToExcel(worksheet, col, result: dict):
    for i, v in enumerate(result.values()):
        worksheet.write(col, i, v)

def packParams(Sp, Se, Serr, Rt, P, R, M_P, M_R):
    return {
        'SIFT_peak': Sp,
        'SIFT_edge': Se,
        'S_err': Serr,
        'Ratio': Rt,
        'Precision': P,
        'Recall': R,
        'M_P': M_P,
        'M_R': M_R,
            }

if __name__ == '__main__':
    gen = MaskedVideoTester()
    SIFT_peak = ParameterRange(0.03, 0.09, 0.015)
    SIFT_edge = ParameterRange(8, 10, 1.0)
    S_err = ParameterRange(0.1, 0.9, 0.2)
    Ratio = ParameterRange(0.7, 0.8, 0.1)
    results = []

    workbook = xlsxwriter.Workbook('results_' + gen.vid_name_orig + '.xlsx')
    worksheet = workbook.add_worksheet()
    printTitle(worksheet)

    for Sp in [0.03, 0.06, 0.09]:
        for Se in [8, 10]:
            for Sf in [15, 17, 20]:
                for Serr in [0.02, 0.1, 0.5]:
                    for Rt in [0.7, 0.8]:
                        P_t, R_t = ([], [])
                        kwargs = {'SIFT_peak': Sp,
                                  'SIFT_edge': Se,
                                  'SIFT_feature_elimination': Sf,
                                  'Samson_err': Serr,
                                  'Ratio': Rt,
                                  'Avg_Precision': 0,
                                  'Avg_Recall': 0,
                                  'M_Precision': 0,
                                  'M_Recall': 0,
                                  }
                        try:
                            slvr = Solver(**kwargs)
                            P_t, R_t = gen.main(slvr)
                        except:
                            pass
                        if len(P_t) * len(R_t) == 0:
                            print(Sp, Se, Sf, Serr, Rt, 'ERROR')
                        else:
                            kwargs['Avg_Precision'] = sum(P_t) / len(P_t)
                            kwargs['Avg_Recall'] = sum(R_t) / len(R_t)
                            kwargs['M_Precision'] = my_median(P_t)
                            kwargs['M_Recall'] = my_median(R_t)
                            print(Sp, Se, Sf, Serr, Rt, 'DONE')
                        results.append(kwargs)

    for i, r in enumerate(results, start=1):
        printToExcel(worksheet, i, r)
    workbook.close()