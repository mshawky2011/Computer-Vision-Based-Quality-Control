import cv2
import numpy as np

cap = cv2.imread('<path for img>', cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

lower = np.array([40,20,0])
upper = np.array([255,255,255])

mask = cv2.inRange(hsv, lower, upper)

mask_inv = cv2.bitwise_not(mask)
res = cv2.bitwise_and(cap, cap, mask = mask)

params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.filterByColor = True

params.filterByCircularity = True
params.minCircularity = 0.5

params.minDistBetweenBlobs = 200

params.minArea = 40
params.filterByArea = True
detector = cv2.SimpleBlobDetector_create(params)
cv2.imshow('orignal', cap)
keypoints = detector.detect(255 - mask)
lenn = len(keypoints)
print(lenn)
# RGB white 255

cv2.imshow('mask', mask) #white pills blsck BG
cv2.imshow('res', res)


cv2.imshow('mask_inv', mask_inv) #black pils and white BG
#cv2.imshow('im_with_keypoints', im_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
