##inputs are : 
# img -> array of cropped image
# points -> xmin,ymin and xmax,ymax
 
import cv2
import numpy as np
import os
from rembg import remove

def get_crop(img,x,y,w,h):#img is a np array
   return img[int(y):int(h),int(x):int(w)]

def foreground(img):
    return remove(img,only_mask=True)

def get_center(x1,y1,x2,y2):
   return (int((x1+x2)/2),int((y1+y2)/2))

def get_rotation(img):
  if img is None:
    print("Error: File not found")
    exit(0)
  
  # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  
  for _,c in enumerate(contours):
    rect = cv2.minAreaRect(c)
  return rect


def postprocessing(img): 
    crop_mask=foreground(img)
    crop_bbox=get_rotation(crop_mask) #((center of rect),(width,height),angle of rotation)

    return crop_bbox[1],crop_bbox[2]

# rect = postprocessing(img,points)
# box=cv2.boxPoints(rect)
# box = np.intp(box)
# cv2.drawContours(original_img,[box],0,(0,0,255),2)
# cv2.imshow("test",original_img)