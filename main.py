# This is a sample Python script.
import cv2 as cv
from matplotlib import pyplot as plt


def loading_displaying_saving():
    img = cv.imread('girl.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow('girl', img)
    cv.waitKey(0)
    #cv2.imwrite('graygirl.jpg', img)

if __name__ == '__main__':
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

