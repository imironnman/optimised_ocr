import sys
import imutils
import numpy as np
import cv2
import argparse
from collections import defaultdict
import os


os.system('clear')

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the Image")
args=vars(ap.parse_args())



def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]



L=[]
dat=[]
dis=[]
spa=[]
k=0 

print('\t##############################WELCOME#########################')
choice=input('Enter the Type of Detection \n  1.Number Detection \n  2.Text Detection \n  3.Text and Number Detection \n  0.Exit  \n\n  Enter the Choice:')

if int(choice)>3 and int(choice) <0:
  print('\n Please enter Valid choice')
  exit()

if int(choice)==0:
  exit()

os.system('clear')

print('\t##############################WELCOME#########################')
choice2=input('Enter the Type of Operation \n  1.Simple Method \n  2.Advanced \n  0.Exit  \n\n  Enter the Choice:')

if int(choice2)>2 and int(choice2) <0:
  print('\n Please enter Valid choice')
  exit()

if int(choice2)==0:
  exit()

if int(choice2)==1:

 image = cv2.imread(args["image"])
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 blur = cv2.GaussianBlur(gray,(5,5),0)
 thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

# show the original and thresholded images
#cv2.imshow("Original", image)
#cv2.imshow("Thresh", thresh)

# find external contours in the thresholded image and allocate memory
# for the convex hull image
 (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
 hullImage = np.zeros(gray.shape[:2], dtype="uint8")
 cnts.sort(key=lambda x:get_contour_precedence(x, image.shape[1]))

 for (i, c) in enumerate(cnts):
	# compute the area of the contour along with the bounding box
	# to compute the aspect ratio

     if cv2.contourArea(c)>50:
        [x,y,w,h] = cv2.boundingRect(c)
        L.append(h)


 key=[]
 ar=[]
 sol=[]
 ext=[]
     

 d = defaultdict(int)
 for i in L:
     d[i] += 1
 result = max(d.items(), key=lambda x: x[1])

 hei,noh=result

 te=sum(L) / len(L)

 fin=hei/te
 fin1=1-fin
 res=hei+fin1*(te-hei)

#print(sum(L) / len(L))
#print(hei+fin1*(te-hei),hei,te)

 if res-te<0:
      ch=te-res-4
 else:
      ch=res-te-4

 value="Fail"
 clone = image.copy()
# loop over the contours
 for (i, c) in enumerate(cnts):
	# compute the area of the contour along with the bounding box
	# to compute the aspect ratio

     if cv2.contourArea(c)>50:
        [x,y,w,h] = cv2.boundingRect(c)
       
#h>res-(ch)/2 and h<res+(ch-5)/2
       #print(h,w)

        if  h>hei-5 and h<hei+5  :

         ellipse = cv2.fitEllipse(c)
         cv2.ellipse(clone, ellipse, (0, 255, 0), 2)

        #cv2.imshow("Bounding Boxes", clone)
        #cv2.waitKey(0) 

         peri=cv2.arcLength(c,True)
        
         approx=cv2.approxPolyDP(c,0.04*peri,True)

        #print(len(approx),peri)


         area = cv2.contourArea(c)

	# compute the aspect ratio of the contour, which is simply the width
	# divided by the height of the bounding box
         aspectRatio = w / float(h)

	# use the area of the contour and the bounding box area to compute
	# the extent
         extent = area / float(w * h)

	# compute the convex hull of the contour, then use the area of the
 	# original contour and the area of the convex hull to compute the
 	# solidity
         hull = cv2.convexHull(c)
         hullArea = cv2.contourArea(hull)
         solidity = area / float(hullArea)

	# visualize the original contours and the convex hull and initialize
	 # the name of the shape
         cv2.drawContours(hullImage, [hull], -1, 255, -1)
         cv2.drawContours(image, [c], -1, (240, 0, 159), 3)
         shape = ""
 
         if aspectRatio >=0.69 and aspectRatio<=0.73 and extent >=0.25 and extent <=0.30 and solidity >=0.43 and solidity<=0.48:
                 value="T"
         elif aspectRatio >=0.14 and aspectRatio<=0.19 and extent >=0.81 and extent <=0.88 and solidity >=0.99 and solidity<=1:
                 value="I"
         elif aspectRatio >=0.67 and aspectRatio<=0.72 and extent >=0.58 and extent <=0.62 and solidity >=0.82 and solidity<=0.86:
                 value="P"
         elif aspectRatio >=0.77 and aspectRatio<=0.80 and extent >=0.74 and extent <=0.78 and solidity >=0.96 and solidity<=0.99:
                 value="O"
         elif aspectRatio >=0.74 and aspectRatio<=0.78 and extent >=0.39 and extent <=0.43 and solidity >=0.45 and solidity<=0.51:
                 value="S"
         elif aspectRatio >=0.56 and aspectRatio<=0.58 and extent >=0.42 and extent <=0.44 and solidity >=0.54 and solidity<=0.56:
                 value="7"
         elif aspectRatio >=0.54 and aspectRatio<=0.56 and extent >=0.48 and extent <=0.50 and solidity >=0.58 and solidity<=0.60:
                 value="9"
         elif aspectRatio >=0.56 and aspectRatio<=0.58 and extent >=0.77 and extent <=0.79 and solidity >=0.90 and solidity<=0.92:
                value="8"
         elif aspectRatio >=0.56 and aspectRatio<=0.58 and extent >=0.70 and extent <=0.72 and solidity >=0.86 and solidity<=0.88:
                 value="6"
         elif aspectRatio >=0.53 and aspectRatio<=0.55 and extent >=0.49 and extent <=0.51 and solidity >=0.59 and solidity<=0.61:
                 value="5"
         elif aspectRatio >=0.57 and aspectRatio<=0.59 and extent >=0.51 and extent <=0.53 and solidity >=0.78 and solidity<=0.80:
                 value="4"
         elif aspectRatio >=0.53 and aspectRatio<=0.55 and extent >=0.53 and extent <=0.55 and solidity >=0.61 and solidity<=0.63:
                 value="3"
         elif aspectRatio >=0.57 and aspectRatio<=0.59 and extent >=0.51 and extent <=0.53 and solidity >=0.56 and solidity<=0.58:
                 value="2"
         elif aspectRatio >=0.38 and aspectRatio<=0.40 and extent >=0.55 and extent <=0.57 and solidity >=0.70 and solidity<=0.72:
                 value="1"        
         elif aspectRatio >=0.55 and aspectRatio<=0.57 and extent >=0.78 and extent <=0.80 and solidity >=0.98 and solidity<=1.00:
                 value="0"   
        
        
	
	# if the aspect ratio is approximately one, then the shape is a squar
         #print(value)

	# draw the shape name on the image
         if int(choice)==1:
          cv2.putText(image, value, (x-5, y +25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(240, 0, 159), 2)
         else:
          cv2.putText(image, value, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(240, 0, 159), 2)
	# show the contour properties
         #print ("Contour #%d -- aspect_ratio=%.2f, extent=%.2f, solidity=%.2f" % (i + 1, aspectRatio, extent, solidity))
        #print(len(c))
       
	# show the output images
        #cv2.imshow("Convex Hull", hullImage)
         cv2.imshow("Image", image)
         cv2.waitKey(0)
 exit()
print('\t##############################WELCOME#########################')
choice1=input('Enter the Type of Operation \n  1.Training \n  2.Testing \n  0.Exit  \n\n  Enter the Choice:')
#print(choice)

if int(choice1)>2 and int(choice1) <0:
  print('\n Please enter Valid choice')
  exit()

if int(choice1)==0:
  exit()

im = cv2.imread(args["image"])
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)


thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)

if int(choice) ==1:
  _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
elif int(choice)==2:
  _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
elif int(choice)==3:
 _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

hullImage = np.zeros(gray.shape[:2], dtype="uint8")

cnts.sort(key=lambda x:get_contour_precedence(x, im.shape[1]))


for (i, c) in enumerate(cnts):


    if cv2.contourArea(c)>50:
       [x,y,w,h] = cv2.boundingRect(c)
       L.append(h)


       

d = defaultdict(int)
for i in L:
    d[i] += 1
result = max(d.items(), key=lambda x: x[1])

hei,noh=result

te=sum(L) / len(L)

fin=hei/te
fin1=1-fin
res=hei+fin1*(te-hei)

#print(sum(L) / len(L))
#print(hei+fin1*(te-hei),hei,te)

if res-te<0:
     ch=te-res-4
else:
     ch=res-te-4

if int(choice1)==1:

 samples =  np.empty((0,100))
 responses = []
 keys = [i for i in range(48,122)]

 for (i, c) in enumerate(cnts):
     if cv2.contourArea(c)>50:
         [x,y,w,h] = cv2.boundingRect(c)



         #print(h)
#h>res-(ch)/2 and h<res+(ch-5)/2
         if  h>hei-5 and h<hei+5 :
             cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
             roi = thresh[y:y+h,x:x+w]
             roismall = cv2.resize(roi,(10,10))
             cv2.imshow('norm',im)

             k+=1
             if k!=1:
              ele = dis.pop()
              val=x-ele
              print(val)
              spa.append(val)
             dis.append(x)

             key = cv2.waitKey(0)
             #print(x)

             if key == 27:  
                 sys.exit()
             elif key in keys:
                 responses.append(int((key)))
                 sample = roismall.reshape((1,100))
                 samples = np.append(samples,sample,0)

 responses = np.array(responses,np.float32)
 responses = responses.reshape((responses.size,1))
 print ("Training completed Successfully")
 
 if int(choice)==1:
  np.savetxt('generalsamples.data',samples)
  np.savetxt('generalresponses.data',responses)
 elif int(choice)==2:
  np.savetxt('generalsamples1.data',samples)
  np.savetxt('generalresponses1.data',responses)
 else:
  np.savetxt('generalsamples2.data',samples)
  np.savetxt('generalresponses2.data',responses)

if int(choice1)==2:

 out = np.zeros(im.shape,np.uint8)

 if int(choice)==1:
  samples = np.loadtxt('generalsamples.data',np.float32)
  responses = np.loadtxt('generalresponses.data',np.float32)
 elif int(choice)==2:
  samples = np.loadtxt('generalsamples1.data',np.float32)
  responses = np.loadtxt('generalresponses1.data',np.float32)
 else:
  samples = np.loadtxt('generalsamples2.data',np.float32)
  responses = np.loadtxt('generalresponses2.data',np.float32)
 responses = responses.reshape((responses.size,1))

 model = cv2.ml.KNearest_create()
 model.train(samples,cv2.ml.ROW_SAMPLE,responses)


 val="FAILED"


 for (i, c) in enumerate(cnts):
     if cv2.contourArea(c)>50:
         [x,y,w,h] = cv2.boundingRect(c)




         if  h>=hei-5 and h<=hei+5:
             cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
             roi = thresh[y:y+h,x:x+w]
             roismall = cv2.resize(roi,(10,10))
             roismall = roismall.reshape((1,100))
             roismall = np.float32(roismall)
 
             k+=1
             if k!=1:
              ele = dis.pop()
              val=x-ele
              spa.append(val)
             elif k==1:
              spa.append(x)
         #print(h)

             dis.append(x)


             retval, results, neigh_resp, dists = model.findNearest(roismall, k = 3)
             string = str(int((results)))

             if string == "97":
               val="A"
             if string == "98":
               val="B"
             if string == "99":
               val="C"
             if string == "100":
               val="D"
             if string == "101":
               val="E"
             if string == "102":
               val="F"
             if string == "103":
               val="G"
             if string == "104":
               val="H"
             if string == "105":
               val="I"
             if string == "106":
               val="J"
             if string == "107":
               val="K"
             if string == "108":
               val="L"
             if string == "109":
               val="M"
             if string == "110":
               val="N"
             if string == "111":
               val="O"
             if string == "112":
               val="P"
             if string == "113":
               val="Q"
             if string == "114":
               val="R"
             if string == "115":
               val="S"
             if string == "116":
               val="T"
             if string == "117":
               val="U"
             if string == "118":
               val="V"
             if string == "119":
               val="W"
             if string == "120":
               val="X"
             if string == "121":
               val="Y"
             if string == "122":
               val="Z"
             if string == "48":
               val="0"
             if string == "49":
               val="1"
             if string == "50":
               val="2"
             if string == "51":
               val="3"
             if string == "52":
               val="4"
             if string == "53":
               val="5"
             if string == "54":
               val="6"
             if string == "55":
               val="7"
             if string == "56":
               val="8"
             if string == "57":
               val="9"
             #print (val)
             dat.append(val)
             cv2.putText(out,val,(x,y+h),0,1,(0,255,0))

 cv2.imshow('im',im)
 cv2.imshow('out',out)
 #print (out)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
 
 valav=spa.copy()
 repno=spa.copy()

 e = defaultdict(int)
 for i in repno:
     e[i] += 1
 result1 = max(e.items(), key=lambda x: x[1])

 hei1,noh1=result1

 print(hei1,noh1)


 for i in range(len(valav)):
  if valav[i]<0:
   
    valav[i]=0
 #print(valav,i)
 zi=sum(valav) / len(valav)
 print(spa)
 f = open('finop.txt', 'w')
 for i in range(len(spa)):
   #print (i)
   if spa[i]<-200:
    f.write("\n%s"%dat[i])
   elif spa[i]<zi or i==0:
    f.write("%s" % (dat[i]))
   else:
    f.write("\t %s"%(dat[i]))
 f.close()
print('Thank u')
