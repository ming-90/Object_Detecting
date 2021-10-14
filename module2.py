
import cv2
from pylibdmtx.pylibdmtx import decode
import time


# 리사이징 해서 검출 하는 방법
# 사진에 바코드가 있으면 검출이 금방 끝나지만 없으면 엄청 오래 걸림
# 바코드 한개의 크기가 125X125 정도의 크기면 바코드가 없어도 느리지 않게 검출 가능
# 특정 크기의 이미지를 받아서 바코드 위치별로 리사이징 하고 거기서 디텍팅

def mainImgRead(im):
    #startHeight = [640,870]
    startHeight = [0,280]
    startWidth = [100,380]
    plusWidth = 700
    plusHeight = 357
    for j in range(9):
        heisize = plusHeight * j
        bigIm = im[640+heisize:890+heisize,0:-1]
        for i in range(4):
            size = plusWidth * i
            resizeImg = bigIm[startHeight[0]:startHeight[1], startWidth[0] + size:startWidth[1] + size]
            resize = testresize(resizeImg)
            #cv2.imwrite('./testImg/test' + str(j) + '_' + str(i) + '.jpg', resize)
            decodData = decode(resize)
            #print(decodData)
           

def testImgRead(im):
    size = 357
    for i in range(9):
        imgsize = size * i
        bigIm = im[635 + imgsize:890 + imgsize,0:-1]
        cv2.imwrite('./rowImg/row' + str(i) + '.jpg', bigIm)
        decodData = decode(bigIm)
        print(decodData)


def testresize(img):
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100) 
    height = int(img.shape[0] * scale_percent / 100) 
    dim = (width, height) 
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    #cv2.imwrite('./resize.jpg', resized)
    return resized


def testWebCam(im):
    #im = cv2.imread('barcode5.jpg', cv2.IMREAD_GRAYSCALE)
    decodData = decode(im)
    print(decodData)
    w, h, _ = im.shape
    print(w, h)

def imageBinary(im):
    gray_tr01 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    border01, binary01 = cv2.threshold(gray_tr01, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./binary.jpg', border01)
    cv2.imwrite('./binary.jpg', binary01)


if __name__ == '__main__':
    im = cv2.imread('barcode3.jpg', cv2.IMREAD_COLOR)
    startTime = time.time()
    mainImgRead(im)
    #imageBinary(im)
    #testImgRead()
    #testImgRead2()
    #testresize(im)
    #mainImgRead(resized)
    #testImgRead(resized)
    #resize = testresize(im)
    #testWebCam(resize)
    endTime = time.time()
    print(endTime - startTime)
