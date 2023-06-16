# This is a sample Python script.
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from VideoHandler import *


if __name__ == '__main__':
    VH = VideoHandler()
    vid_name = 'InputVideo.mp4'
    VH.selectSpecularArea(vid_name)
