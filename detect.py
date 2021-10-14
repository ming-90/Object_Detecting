# import the necessary packages
import numpy as np
import cv2
import argparse
from pylibdmtx.pylibdmtx import decode
import time, os
from multiprocessing import Pool

screenRate = 0.3
detectPadding = 6

def detect(downimage):
    #image = cv2.resize(image, dsize=(1000, 1000), interpolation=cv2.INTER_AREA)
    global image 
    image = cv2.resize(downimage, dsize=(0, 0), fx=screenRate, fy=screenRate, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cv2.imwrite('testImg/gray.jpg', gray)

    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    #cv2.imwrite('testImg/blurred.jpg', blurred)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('testImg/closed.jpg', closed)

    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)

    b,g,r,a = 0,0,255,0

    #멀티 프로세싱
    p = Pool(5)
    p.map(drawLine, c)
    #drawLine(c)

    return image 

def drawLine(i):
    time.sleep(1000)
    rect = cv2.minAreaRect(i)
    box = np.int0(cv2.boxPoints(rect))
    x, y, w, h = cv2.boundingRect(i)
    if x == 0 or y == 0 or w == 0 or h == 0: return 
    y1 = (y-detectPadding if y - detectPadding > 0 else 0)
    x1 = (x-detectPadding if x - detectPadding > 0 else 0)
    cv2.imwrite('testImg/test' + str(count) + '.jpg', image[y1:y+h+detectPadding,x1:x+w+detectPadding])
    #barcode = decode(image[y1:y+h+detectPadding,x1:x+w+detectPadding])
    #if barcode == []: continue
    #if barcode == []: Pbarcode = ''
    #else: Pbarcode = str(barcode[0].data)
    #cv2.putText(image,  Pbarcode, (box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_8)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 1)

    cv2.imwrite('testImg/1.jpg', image)



def Image():
    image = cv2.imread('sample/barcode3.jpg')
    showImg = detect(image)
    cv2.imshow('',showImg)
    key = cv2.waitKey(1)


def webcam():
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == 0: print("열수 없음")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    i = 0
    while(cap.isOpened()):
        ret, image = cap.read()
        if not ret: continue
        showImg = detect(image)

        cv2.imshow('',showImg)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        elif key == ord('s'): 
            i += 1
            #cv2.imwrite('c_%03d.jpg' % i, showImg)


def webcamOnce():
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == 0: print("열수 없음")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    i = 0
    if cap.isOpened():
        ret, image = cap.read()
        showImg = detect(image)

#start_time = time.time()
Image()
#webcam()
#webcamOnce()
#end_time = time.time()
#print(end_time - start_time)





