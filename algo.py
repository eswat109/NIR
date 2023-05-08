import numpy as np
from numpy.linalg import norm
import cv2 as cv
from matplotlib import pyplot as plt
from enum import Enum


class Solver:

    # '''ARTICLE PARAMETERS'''
    SIFT_peak_threshold = 0.07 #0.08
    SIFT_edge_threshold = 10

    SIFT_feature_elimination_threshold = 25
    SIFT_matching_threshold = 2

    samson_err = 0.3 #0.5

    RANSAC_nIter = 2000

    ratio_test = 0.7

    Gaussian_kernel = 61
    Gaussian_deviation = 0
    Specular_thr = 0.1

    blur_on = True
    g_blur_on = True
    sized_on = False
    intens_on = True
    is_thrs_on = False
    point_size = 10
    ED_size = point_size #30
    AD_size = point_size  #

    class AndMethod(Enum):
        MUL = 0
        CONT_MUL = 1
        MIN = 2
        MAX = 3

    andMethod = AndMethod.MIN

    #def __init__(self, SIFT_peak=0.09, SIFT_edge=8, Samson_err=0.5, Ratio=0.8, G_deviation=0, P_size=1):
    def __init__(self, **kwargs):
        if 'SIFT_peak' in kwargs.keys():
            self.SIFT_peak_threshold = kwargs['SIFT_peak']
        if 'SIFT_edge' in kwargs.keys():
            self.SIFT_edge_threshold = kwargs['SIFT_edge']
        if 'SIFT_feature_elimination' in kwargs.keys():
            self.SIFT_feature_elimination_threshold = kwargs['SIFT_feature_elimination']
        if 'Samson_err' in kwargs.keys():
            self.samson_err = kwargs['Samson_err']
        if 'Ratio' in kwargs.keys():
            self.ratio_test = kwargs['Ratio']
        if 'Gaussian_deviation' in kwargs.keys():
            self.SIFT_edge_threshold = kwargs['G_deviation']
        if 'P_size' in kwargs.keys():
            self.point_size = kwargs['P_size']

    def makeFieldByPoints(self, img, pts, intns = None):
        blank_image = np.zeros(img.shape, np.uint8)
        maxIntns = 255
        for i, pt in enumerate(pts):
            color = 1
            if self.intens_on:
                if intns is not None:
                    color = intns[i]
            blank_image = cv.circle(blank_image, tuple(pt), self.point_size, color * maxIntns, -1)
        if not self.blur_on:
            return blank_image
        if self.g_blur_on:
            return cv.GaussianBlur(blank_image, (self.Gaussian_kernel, self.Gaussian_kernel), self.Gaussian_deviation)
        return cv.blur(blank_image, (self.Gaussian_kernel, self.Gaussian_kernel), self.Gaussian_deviation)

    def makeImgFromField(self, field):
        maxIntns = 255
        return np.multiply(maxIntns, field)

    def makeFieldFromIg(self, field):
        maxIntns = 255
        return np.divide(field, maxIntns)

    def andMasks(self, m1, m2):
        '''
        if self.is_thrs_on:
            _, m1 = cv.threshold(m1, int(self.Specular_thr * 255), 255, cv.THRESH_BINARY)
            _, m2 = cv.threshold(m2, int(self.Specular_thr * 255), 255, cv.THRESH_BINARY)
        '''
        m = np.zeros(m1.shape)
        if self.andMethod == self.AndMethod.MUL:
            m = cv.multiply(m1, m2)
        elif self.andMethod == self.AndMethod.MAX:
            m = np.maximum(m1, m2)
        elif self.andMethod == self.AndMethod.MIN:
            m = np.minimum(m1, m2)
        elif self.andMethod == self.AndMethod.CONT_MUL:
            r1 = m1.astype(float) / 255
            r2 = m2.astype(float) / 255
            r = np.multiply(r1, r2)
            m = r * 255
            m = m.astype(np.uint8)

        if self.is_thrs_on:
            _, m = cv.threshold(m, int(self.Specular_thr * 255), 255, cv.THRESH_BINARY)

        return m

    def getMasksFromFrames(self, img1, img2):
        ''' MAKE SPECULAR MASK FROM TWO CONSISTENT FRAMES '''

        # SIFT CREATE
        sift = cv.SIFT_create(contrastThreshold=self.SIFT_peak_threshold, edgeThreshold=self.SIFT_edge_threshold)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # AVERAGE VECTOR THRESHOLD
        if des1 is not None:
            des1 = [des for des in des1 if np.average(des) > self.SIFT_feature_elimination_threshold]
            des1 = np.array(des1)
        if des2 is not None:
            des2 = [des for des in des2 if np.average(des) > self.SIFT_feature_elimination_threshold]
            des2 = np.array(des2)
            #av1 = [np.average(des) for des in des1]
            #av1 = [a for a in av1 if a > self.SIFT_feature_elimination_threshold]
            '''flag = 0
            for d in av1:
                if d <= self.SIFT_feature_elimination_threshold:
                    flag += 1
            if flag > 0:
                print(flag)'''

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # KDTREE MATCHING
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=self.SIFT_matching_threshold)
        pts1 = []
        pts2 = []
        kpts1 = []
        kpts2 = []
        dess1 = []
        dess2 = []
        good = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                kpts2.append(kp2[m.trainIdx])
                kpts1.append(kp1[m.queryIdx])
                dess2.append(des2[m.trainIdx])
                dess1.append(des1[m.queryIdx])
        dess1 = np.array(dess1)
        dess2 = np.array(dess2)
        kpts1 = np.array(kpts1)
        kpts2 = np.array(kpts2)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)


        # FUNDAMENTAL MATRIX CALCULATION
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC, ransacReprojThreshold=self.samson_err, confidence=0.99,
                                        maxIters=self.RANSAC_nIter)
        # INLINERS
        inl_pts1 = pts1[mask.ravel() == 1]
        inl_pts2 = pts2[mask.ravel() == 1]
        inl_des1 = []
        inl_des2 = []
        inl_kpts1 = []
        inl_kpts2 = []
        for i, m in enumerate(mask.ravel()):
            if m == 1:
                inl_des1.append(dess1[i])
                inl_des2.append(dess2[i])
                inl_kpts1.append(kpts1[i])
                inl_kpts2.append(kpts2[i])

        # DESCRIPTOR L1 NORM
        des_dif = np.abs(dess2 - dess1)
        des_l1 = [norm(d, 1) for d in des_dif]
        des_l1 = np.array(des_l1)
        des_l1_norm = ((des_l1 - des_l1.min()) / (des_l1.max() - des_l1.min()))

        # OUTLINERS
        outl_pts1 = pts1[mask.ravel() == 0]
        outl_pts2 = pts2[mask.ravel() == 0]
        outl_kpts1 = []
        outl_kpts2 = []
        for i, m in enumerate(mask.ravel()):
            if m == 0:
                outl_kpts1.append(kpts1[i])
                outl_kpts2.append(kpts2[i])

        ED_mask = self.makeFieldByPoints(img2, outl_pts2)
        AD_mask = self.makeFieldByPoints(img2, inl_pts2, des_l1_norm)

        SPEC_mask = self.andMasks(ED_mask, AD_mask)

        '''if is_thrs_on:
            _, SPEC_mask = cv.threshold(SPEC_mask, Specular_thr, 255, cv.THRESH_TOZERO)'''

        return (SPEC_mask, ED_mask, AD_mask)

    def combineFrameAndMask(self, img, mask):
        #color = np.array([255, 0, 255])
        #white_mask = cv.cvtColor(makeImgFromField(mask), cv.COLOR_GRAY2RGB)
        white_mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        r, g, b = cv.split(white_mask)
        z = np.zeros(g.shape, np.uint8)
        color_mask = cv.merge([r, z, b])
        res = cv.addWeighted(img, 1, color_mask, 0.7, 0.0)
        return res

    def getMaskedImage(self, img1, img2):
        SPEC_mask, ED_mask, AD_mask = self.getMasksFromFrames(img1, img2)
        img_mask = self.combineFrameAndMask(img2_2, SPEC_mask)
        return img_mask

    def getMaskArea(self, mask):
        contours, hierarchy = cv.findContours(image=mask, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)
        totalArea = 0.0
        for c in contours:
            totalArea += cv.contourArea(c)
        return totalArea

    def getPrecisionRecall(self, maskSpec, maskTrue):
        maskIntersec = cv.bitwise_and(maskSpec, maskTrue)
        '''cv.imshow('img', maskTrue)
        cv.waitKey(0)'''
        I = self.getMaskArea(maskIntersec)
        D = self.getMaskArea(maskSpec)
        G = self.getMaskArea(maskTrue)
        Precision = I / D
        Recall = I / G
        return (Precision, Recall)

    def combineImgAndMasks(self, img, ED, AD, SPEC):
        pass


if __name__ == '__main__':
    img_path = 'img/RealTestset2013/Office/'
    img_name1 = '20131112_152921.jpg'
    img_name2 = '20131112_152924.jpg'
    img_path = 'img/RealTestset2013/Hallway/'
    img_name1 = '20131114_154749.jpg'
    img_name2 = '20131114_154753.jpg'
    img_path = 'img/RealTestset2013/Street/'
    img_name1 = '20131114_155630.jpg'
    img_name2 = '20131114_155632.jpg'
    img_path = 'img/RealTestset2013/Cabinet/'
    img_name1 = 'Cabinet2_Full_1.jpg'
    img_name2 = 'Cabinet2_Full_2.jpg'
    img_name1 = 'Cabinet3_F_Sp_1.jpg'
    img_name2 = 'Cabinet3_F_Sp_2.jpg'
    '''
    '''
    kwargs = {'SIFT_peak': 0.09,
              'SIFT_edge': 8,
              'SIFT_feature_elimination': 20,
              'Samson_err': 0.02,
              'Ratio': 0.7,
              }
    slvr = Solver(**kwargs)

    img1 = cv.imread(img_path + img_name1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img_path + img_name2, cv.IMREAD_GRAYSCALE)
    img1_2 = cv.imread(img_path + img_name1, cv.IMREAD_COLOR)
    img2_2 = cv.imread(img_path + img_name2, cv.IMREAD_COLOR)

    SPEC_mask, ED_mask, AD_mask = slvr.getMasksFromFrames(img1, img2)

    plt.subplot(221), plt.imshow(img2_2)
    plt.subplot(222), plt.imshow(SPEC_mask)
    #plt.subplot(222), plt.imshow(slvr.combineFrameAndMask(img2_2, SPEC_mask))
    #plt.subplot(142), plt.imshow(slvr.combineFrameAndMask(img2_2, ED_mask))
    #plt.subplot(143), plt.imshow(slvr.combineFrameAndMask(img2_2, AD_mask))
    plt.subplot(223), plt.imshow(ED_mask)
    plt.subplot(224), plt.imshow(AD_mask)



    '''
    intsec = cv.bitwise_and(SPEC_mask, img3)
    plt.subplot(241), plt.imshow(ED_mask)
    plt.subplot(242), plt.imshow(AD_mask)
    plt.subplot(243), plt.imshow(SPEC_mask)

    plt.subplot(245), plt.imshow(img2_2)
    plt.subplot(246), plt.imshow(SPEC_mask)
    plt.subplot(247), plt.imshow(img3)
    plt.subplot(248), plt.imshow(intsec)
    I = slvr.getMaskArea(intsec)
    D = slvr.getMaskArea(SPEC_mask)
    G = slvr.getMaskArea(img3)
    print (I, D, G)
    '''
    plt.show()
