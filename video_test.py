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

    def main(self, SIFT_peak=0.08, SIFT_edge=10, Samson_err=0.5, Ratio=0.7, G_deviation=30, P_size=30):

        slvr = Solver(SIFT_peak, SIFT_edge, Samson_err, Ratio, G_deviation, P_size)

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
    worksheet.write(0, 0, 'SIFT_peak')
    worksheet.write(0, 1, 'SIFT_edge')
    worksheet.write(0, 2, 'Sampson_err')
    worksheet.write(0, 3, 'Ratio')
    worksheet.write(0, 4, 'Avg Precision')
    worksheet.write(0, 5, 'Avg Recall')
    worksheet.write(0, 6, 'M Precision')
    worksheet.write(0, 7, 'M Recall')

def printToExcel(worksheet, col, result):
    worksheet.write(col, 0, result['SIFT_peak'])
    worksheet.write(col, 1, result['SIFT_edge'])
    worksheet.write(col, 2, result['S_err'])
    worksheet.write(col, 3, result['Ratio'])
    worksheet.write(col, 4, result['Precision'])
    worksheet.write(col, 5, result['Recall'])
    worksheet.write(col, 6, result['M_P'])
    worksheet.write(col, 7, result['M_R'])

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

    for Sp in np.arange(SIFT_peak.from_, SIFT_peak.to_ + (SIFT_peak.step_ / 2), SIFT_peak.step_):
        for Se in np.arange(SIFT_edge.from_, SIFT_edge.to_ + (SIFT_edge.step_ / 2), SIFT_edge.step_):
            for Serr in np.arange(S_err.from_, S_err.to_ + (S_err.step_ / 2), S_err.step_):
                for Rt in np.arange(Ratio.from_, Ratio.to_ + (Ratio.step_ / 2), Ratio.step_):
                    P_t, R_t = ([], [])
                    try:
                        P_t, R_t = gen.main(SIFT_peak=Sp, SIFT_edge=Se, Samson_err=Serr, Ratio=Rt, G_deviation=30, P_size=20)
                    except:
                        pass
                    if len(P_t) * len(R_t) == 0:
                        results.append(packParams(Sp, Se, Serr, Rt, 0, 0, 0, 0))
                        print(Sp, Se, Serr, Rt, 'ERROR')
                    else:
                        P = sum(P_t) / len(P_t)
                        R = sum(R_t) / len(R_t)
                        results.append(packParams(Sp, Se, Serr, Rt, P, R, my_median(P_t), my_median(R_t)))
                        print(Sp, Se, Serr, Rt, 'DONE')

    for i, r in enumerate(results, start=1):
        printToExcel(worksheet, i, r)
    workbook.close()