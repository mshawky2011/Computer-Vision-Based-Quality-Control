import cv2
import numpy as np

cap = cv2.imread('/media/parth/06C20E27C20E1B95/machine leaning/sentdex/opencv/medicine.jpg', cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

lower = np.array([40,20,0])
upper = np.array([255,255,255])

mask = cv2.inRange(hsv, lower, upper)

mask_inv = cv2.bitwise_not(mask)
res = cv2.bitwise_and(cap, cap, mask = mask)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor= 255
params.filterByColor = True


# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(mask_inv)
im_with_keypoints = cv2.drawKeypoints(cap, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(str(keypoints))

cv2.imshow('orignal', cap)
cv2.imshow('mask', mask)
cv2.imshow('res', res)


cv2.imshow('mask_inv', mask_inv)
#cv2.imshow('im_with_keypoints', im_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()