import sys
import imutils
import numpy as np
import cv2
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the Image")
args=vars(ap.parse_args())


im = cv2.imread(args["image"])
im3 = im.copy()

dis=[]
spa=[]
k=0   

#def sort_contours(cnts, method="left-to-right"):
#	# initialize the reverse flag and sort index
#	reverse = False
#	i = 0
#
#	# handle if we need to sort in reverse
#	if method == "right-to-left" or method == "bottom-to-top":
#		reverse = True
#
#	# handle if we are sorting against the y-coordinate rather than
#	# the x-coordinate of the bounding box
#	if method == "top-to-bottom" or method == "bottom-to-top":
#		i = 1

#	# construct the list of bounding boxes and sort them from top to
#	# bottom
#	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
#		key=lambda b:b[1][i], reverse=reverse))

#	# return the list of sorted contours and bounding boxes
#	return (cnts, boundingBoxes)


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    #print(((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0])
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#dilated = cv2.dilate(gray.copy(), None, iterations= 3)
#eroded = cv2.erode(dilated.copy(), None, iterations= 3)

blur = cv2.GaussianBlur(gray.copy(),(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

_,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

cnts.sort(key=lambda x:get_contour_precedence(x, im.shape[1]))


#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

#(cnts, boundingBoxes) = sort_contours(cnts, method="left-to-right")
#(cnts, boundingBoxes) = sort_contours(cnts, method="top-to-bottom")


samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in cnts:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        k+=1
        if k!=1:
         ele = dis.pop()
         val=x-ele
         print(val)
         if val>40:
           spa.append(1)
         elif val<0:
           spa.append(-1)
         else:
           spa.append(0)
        dis.append(x)
        
        origin = cv2.boundingRect(cnt)


        if  h>20 and h<80:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)
            #print(x)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

print(spa)
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print ("training complete")

