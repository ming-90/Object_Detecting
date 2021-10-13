# import the necessary packages
import numpy as np
import cv2
import argparse

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "path to the image file")
# args = vars(ap.parse_args())

def detect(image):
    image = cv2.resize(image, dsize=(1000, 1000), interpolation=cv2.INTER_AREA)
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

    for i in c:
        rect = cv2.minAreaRect(i)
        box = np.int0(cv2.boxPoints(rect))

        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    cv2.imshow('',image)
    cv2.waitKey(0)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv2.imshow('test', frame) 
    k = cv2.waitKey(1) 
    if k == 27: break
