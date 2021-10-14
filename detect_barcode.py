
# import the necessary packages
from pyimagesearch import simple_barcode_detection
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
args = vars(ap.parse_args())
# if the video path was not supplied, grab the reference to the
# camera
if not args.get("video", False):
 camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
 camera = cv2.VideoCapture(args["video"])