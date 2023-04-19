import numpy as np
from numpy.linalg import norm
import cv2 as cv
from matplotlib import pyplot as plt


'''def drawFieldPoints(img, pts):
    color = (0, 255, 0)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for i, pt in enumerate(pts):
        img = cv.circle(img, tuple(pt), 5, color, -1)
    return img

def drawFieldKPoints(img, kpts):
    color = (0, 255, 0)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for i, pt in enumerate(kpts):
        img = cv.circle(img, tuple(pt.pt), pt.size, color, -1)
    return img'''

'''ARTICLE PARAMETERS'''
SIFT_peak_threshold = 0.09 #0.08
SIFT_edge_threshold = 10
SIFT_feature_elimination_threshold = 5
SIFT_matching_threshold = 2
samson_err = 0.3 #0.5
RANSAC_nIter = 2000
ratio_test = 0.7
Gaussian_kernel = 0
Gaussian_deviation = 30
Specular_thr = 0.001

g_blur_on = True
blur_on = True
sized_on = False
intens_on = False
is_thrs_on = True
point_size = 20
ED_size = point_size #30
AD_size = point_size  #

def makeFieldByPoints(img, pts, intns = None):
    blank_image = np.zeros(img.shape, np.uint8)
    for i, pt in enumerate(pts):
        color = 1
        if intens_on:
            if intns is not None:
                color = intns[i]
        blank_image = cv.circle(blank_image, tuple(pt), AD_size, color, -1)
    if not blur_on:
        return blank_image
    if g_blur_on:
        return cv.GaussianBlur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)
    return cv.blur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)

def makeImgFromField(field):
    maxIntns = 255
    return np.multiply(maxIntns, field)

def getMasksFromFrames(img1, img2):
    ''' MAKE SPECULAR MASK FROM TWO CONSISTENT FRAMES '''

    # SIFT CREATE
    sift = cv.SIFT_create(contrastThreshold=SIFT_peak_threshold, edgeThreshold=SIFT_edge_threshold)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # AVERAGE VECTOR THRESHOLD
    '''av1 = [np.average(des) for des in des1]
    flag = 0
    for d in av1:
        if d <= 10:
            flag += 1
    print(flag)'''

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # KDTREE MATCHING
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=SIFT_matching_threshold)
    pts1 = []
    pts2 = []
    kpts1 = []
    kpts2 = []
    dess1 = []
    dess2 = []
    good = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio_test * n.distance:
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
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC, ransacReprojThreshold=samson_err, confidence=0.995,
                                    maxIters=RANSAC_nIter)
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
    _ = 1
    #des_dif = [np.abs(d2 - d1) for (d1, d2) in zip(dess1, dess2)]
    #des_l1 = [norm(d, 1) for d in des_dif]

    # OUTLINERS
    outl_pts1 = pts1[mask.ravel() == 0]
    outl_pts2 = pts2[mask.ravel() == 0]
    outl_kpts1 = []
    outl_kpts2 = []
    for i, m in enumerate(mask.ravel()):
        if m == 0:
            outl_kpts1.append(kpts1[i])
            outl_kpts2.append(kpts2[i])

    '''
    matchesMask = mask.ravel().tolist()
    matchesMask = [1 if not i else 0 for i in matchesMask]
    draw_params = dict(matchColor=None,  # draw matches in green color
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    '''
    # img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # kpts1 = kpts1[mask.ravel() == 0]
    # img3 = cv.drawKeypoints(img1, kpts1, None)
    #img3 = drawFieldPoints(img1, outl_pts1)

    '''
    img3 = makeFieldED(img2, outl_pts2)
    img4 = makeFieldAD(img2, inl_pts2, des_l1_norm)
    if sized_on:
        img3 = makeFieldED_KP(img2, outl_kpts2)
        img4 = makeFieldAD_KP(img2, inl_kpts2, des_l1_norm)

    img3 = cv.cvtColor(img3, cv.COLOR_RGB2GRAY)
    img4 = cv.cvtColor(img4, cv.COLOR_RGB2GRAY)

    if is_thrs_on:
        _, img3 = cv.threshold(img3, Specular_thr, 255, cv.THRESH_BINARY)
        _, img4 = cv.threshold(img4, Specular_thr, 255, cv.THRESH_BINARY)

    #img5 = cv.bitwise_and(img3, img4)
    img5 = cv.multiply(img3, img4)
    plt.subplot(141), plt.imshow(img2_2)
    plt.subplot(142), plt.imshow(img3)
    plt.subplot(143), plt.imshow(img4)
    plt.subplot(144), plt.imshow(img5)
    plt.show()'''

    ED_mask = makeFieldByPoints(img2, outl_pts2)
    AD_mask = makeFieldByPoints(img2, inl_pts2, des_l1_norm)

    SPEC_mask = cv.multiply(ED_mask, AD_mask)

    if is_thrs_on:
        _, SPEC_mask = cv.threshold(SPEC_mask, Specular_thr, 255, cv.THRESH_TOZERO)

    return (ED_mask, AD_mask, SPEC_mask)

def combineFrameAndMask(img, mask):
    #color = np.array([255, 0, 255])
    white_mask = cv.cvtColor(makeImgFromField(mask), cv.COLOR_GRAY2RGB)
    r, g, b = cv.split(white_mask)
    z = np.zeros(g.shape, np.uint8)
    color_mask = cv.merge([r, z, b])
    res = cv.addWeighted(img, 1, color_mask, 0.7, 0.0)
    return res

'''
def makeFieldAD(img, pts, intns):
    #color = (0, 1, 0)
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image = cv.cvtColor(blank_image, cv.COLOR_GRAY2BGR)
    color = np.array([0, 255, 0])
    for i, pt in enumerate(pts):
        i_c = color
        if intens_on:
            i_c = np.multiply(color, intns[i])
        blank_image = cv.circle(blank_image, tuple(pt), AD_size, i_c.tolist() , -1)
    if not blur_on:
        return blank_image
    if g_blur_on:
        return cv.GaussianBlur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)
    return cv.blur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)

def makeFieldAD_KP(img, kpts, intns):
    #color = (0, 1, 0)
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image = cv.cvtColor(blank_image, cv.COLOR_GRAY2BGR)
    color = np.array([0, 255, 0])
    for i, pt in enumerate(kpts):
        i_c = color
        if intens_on:
            i_c = np.multiply(color, intns[i])
        blank_image = cv.circle(blank_image, tuple(np.int32(pt.pt)), np.int32(pt.size/2), i_c.tolist() , -1)
    if not blur_on:
        return blank_image
    if g_blur_on:
        return cv.GaussianBlur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)
    return cv.blur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)

def makeFieldED(img, pts):
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image = cv.cvtColor(blank_image, cv.COLOR_GRAY2BGR)
    color = (0, 255, 0)
    for i, pt in enumerate(pts):
        blank_image = cv.circle(blank_image, tuple(pt), ED_size, color, -1)
    if not blur_on:
        return blank_image
    if g_blur_on:
        return cv.GaussianBlur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)
    return cv.blur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)

def makeFieldED_KP(img, kpts):
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image = cv.cvtColor(blank_image, cv.COLOR_GRAY2BGR)
    color = (0, 255, 0)
    for i, pt in enumerate(kpts):
        blank_image = cv.circle(blank_image, tuple(np.int32(pt.pt)), np.int32(pt.size/2), color, -1)
    if not blur_on:
        return blank_image
    if g_blur_on:
        return cv.GaussianBlur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)
    return cv.blur(blank_image, (Gaussian_kernel, Gaussian_kernel), Gaussian_deviation)
'''

if __name__ == '__main__':
    img_name1 = '20131114_155630.jpg'
    img_name2 = '20131114_155632.jpg'
    img_path = 'img/RealTestset2013/Street/'
    img_name1 = '20131112_152921.jpg'
    img_name2 = '20131112_152924.jpg'
    img_path = 'img/RealTestset2013/Office/'
    img_name1 = '20131114_154749.jpg'
    img_name2 = '20131114_154753.jpg'
    img_path = 'img/RealTestset2013/Hallway/'
    '''
    '''
    img1 = cv.imread(img_path + img_name1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img_path + img_name2, cv.IMREAD_GRAYSCALE)
    img1_2 = cv.imread(img_path + img_name1, cv.IMREAD_COLOR)
    img2_2 = cv.imread(img_path + img_name2, cv.IMREAD_COLOR)

    ED_mask, AD_mask, SPEC_mask = getMasksFromFrames(img1, img2)

    img_mask = combineFrameAndMask(img2_2, SPEC_mask)

    plt.subplot(141), plt.imshow(img2_2)
    plt.subplot(142), plt.imshow(img_mask)
    plt.subplot(143), plt.imshow(combineFrameAndMask(img2_2, ED_mask))
    plt.subplot(144), plt.imshow(combineFrameAndMask(img2_2, AD_mask))
    plt.show()
