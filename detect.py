# import the necessary packages
import numpy as np
import cv2
import argparse
from PIL import Image, ImageDraw, ImageFont
from pylibdmtx.pylibdmtx import decode
import time


def detect(downimage):
    #image = cv2.resize(image, dsize=(1000, 1000), interpolation=cv2.INTER_AREA)
    image = cv2.resize(downimage, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    cutimage = cv2.resize(downimage, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)

    b,g,r,a = 0,0,255,0

    count = 0
    for i in c:
        rect = cv2.minAreaRect(i)
        box = np.int0(cv2.boxPoints(rect))
        x, y, w, h = cv2.boundingRect(i)
        if x == 0 or y == 0 or w == 0 or h == 0: continue
        barcode = decode(cutimage[y-5:y+h+5,x-5:x+w+5])
        if barcode == []: continue
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        cv2.putText(image,  str(barcode[0].data), (box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_8)
        count += 1

    #cv2.imshow('',image)
    # cv2.imshow('',cutimage)
    #cv2.waitKey(0)


start_time = time.time()
image = cv2.imread('sample.jpg')
detect(image)
end_time = time.time()
print(end_time - start_time)