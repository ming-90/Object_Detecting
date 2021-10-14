# import the necessary packages
import numpy as np
import cv2
import argparse
from pylibdmtx.pylibdmtx import decode
import time, os
from multiprocessing import Pool
import multiprocessing

class detectClass(object):
    screenRate = 0.3
    detectPadding = 5

    def __init__(self):
        self.arr = multiprocessing.Manager().list()

    def detect(self, downimage):
        #image = cv2.resize(image, dsize=(1000, 1000), interpolation=cv2.INTER_AREA)
        ImportImage = cv2.resize(downimage, dsize=(0, 0), fx=self.screenRate, fy=self.screenRate, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(ImportImage, cv2.COLOR_BGR2GRAY)

        #cv2.imwrite('testImg/gray.jpg', gray)

        gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
        gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        #cv2.imwrite('testImg/blurred.jpg', blurred)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite('testImg/closed.jpg', closed)

        closed = cv2.erode(closed, None, iterations = 4)
        closed = cv2.dilate(closed, None, iterations = 4)

        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key = cv2.contourArea, reverse = True)

        b,g,r,a = 0,0,255,0

        location = []
        for i in c:
            rect = cv2.minAreaRect(i)
            box = np.int0(cv2.boxPoints(rect))
            x, y, w, h = cv2.boundingRect(i)
            y1 = (y-self.detectPadding if y - self.detectPadding > 0 else 0)
            x1 = (x-self.detectPadding if x - self.detectPadding > 0 else 0)
            y2 = y + h + int(self.detectPadding)
            x2 = x + w + int(self.detectPadding)
            location += [[y1,y2,x1,x2]]
            # cv2.drawContours(ImportImage, [box], -1, (0, 255, 0), 1)

        print("multi process start")
        print(len(location))
        # print(location[:len(location)])
        box1 = location[:len(location)//2]
        box2 = location[len(location)//2:]
        process1 = multiprocessing.Process(target=self.drawLine, args=[gray, box1])
        process2 = multiprocessing.Process(target=self.drawLine, args=[gray, box2])
        process1.start()
        process2.start()
        process1.join()
        process2.join()
            # drawLine(c)

        return ImportImage 

    def drawLine(self, image, location):
        count = 0
        for i in location:
            gettime = time.time()
            barcode = decode(image[i[0]:i[1], i[2]:i[3]])
            cv2.imwrite('testImg/' + str(gettime) + '.jpg', image[i[0]:i[1], i[2]:i[3]])
            count += 1
            if barcode != []: self.arr.append(barcode)


    def Image(self):
        image = cv2.imread('sample/barcode3.jpg')
        showImg = self.detect(image)
        # cv2.imshow('',showImg)
        # key = cv2.waitKey(1)
        cv2.imwrite('testImg/0.jpg', showImg)
        print(self.arr)
        print(len(self.arr))


    def webcam(self):
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


    def webcamOnce(self):
        cap = cv2.VideoCapture(0)
        if cap.isOpened() == 0: print("열수 없음")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        i = 0
        if cap.isOpened():
            ret, image = cap.read()
            showImg = detect(image)


if __name__ == "__main__":
    start_time = time.time()
    a = detectClass()
    a.Image()
    # a.webcam() 
    # a.webcamOnce()
    end_time = time.time()
    print(end_time - start_time)
