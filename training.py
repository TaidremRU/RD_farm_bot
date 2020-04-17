import sys
import os

import numpy as np
import cv2
lists = os.listdir(path="set_training")

samples =  np.empty((0,100))
responses = []
for im in lists:
    img_temp = im
    im = cv2.imread("set_training/" + im)
    im3 = im.copy()

    #gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    #thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  
    #220-210 #190 #225
    if len(image) == 23:
        alfa = 225
    else:
        alfa = 172
    i = 0
    while len(image) > i:
        j = 0
        while len(image[i]) > j: 
            if image[i][j] < alfa:
                image[i][j] = 255
              
            else:
                image[i][j] = 0
            
            j += 1
        i += 1

        
    #cv2.imwrite("set_training/" + img_temp + ".png",image)
    #################      Now finding Contours         ###################
    


        
    contours,hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

   
    keys = [i for i in range(48,58)]
    

    for cnt in contours:
        if cv2.contourArea(cnt)>10 and cv2.contourArea(cnt)<100:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if  h>10:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
                roi = image[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('norm',image)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)
    
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))

print ("training complete")

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
