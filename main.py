# This is a sample Python script.
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def loading_displaying_saving():
    img = cv.imread('girl.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow('girl', img)
    cv.waitKey(0)
    #cv2.imwrite('graygirl.jpg', img)

def temp():
    #loading_displaying_saving()
    img_name1 = '20131114_155630.jpg'
    img_name2 = '20131114_155632.jpg'
    img_path = 'img/RealTestset2013/Street/'
    img_name1 = '20131112_152921.jpg'
    img_name2 = '20131112_152924.jpg'
    img_path = 'img/RealTestset2013/Office/'
    img_name1 = '20131114_154745.jpg'
    img_name2 = '20131114_154749.jpg'
    img_path = 'img/RealTestset2013/Hallway/'
    '''
    '''
    img1 = cv.imread(img_path + img_name1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img_path + img_name2, cv.IMREAD_GRAYSCALE)
    _, img3 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
    cv.imshow('girl', img3)
    cv.waitKey(0)
    plt.subplot(121), plt.imshow(img2)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

def drawCircleExp():
    # create black background
    background = np.zeros((450, 450, 3), np.uint8)

    # initialize the mask of same shape but single channel
    mask = np.zeros((450, 450), np.uint8)

    # draw a circle onto the mask and apply Gaussian blur
    mask = cv.circle(mask, (250, 250), 1, (255, 255, 255), -1, cv.LINE_AA)
    mask1 = cv.GaussianBlur(mask, (61, 61), 0)
    mask2 = cv.GaussianBlur(mask, (0, 0), 30)
    mask3 = cv.GaussianBlur(mask, (61, 61), 30)
    plt.subplot(221), plt.imshow(mask)
    plt.subplot(222), plt.imshow(mask1)
    plt.subplot(223), plt.imshow(mask2)
    plt.subplot(224), plt.imshow(mask3)
    plt.show()

if __name__ == '__main__':
    drawCircleExp()
