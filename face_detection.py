import cv2
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
# from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils


i=0
cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FPS,10)
while True:
    result, img = cam.read()
    cv2.imwrite(f'hello{i}.jpg',img)
    # cv2.waitKey(2)
    i+=1
    time.sleep(2)
