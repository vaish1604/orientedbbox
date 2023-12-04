import cv2
import numpy as np
from functions import get_crop,get_center,postprocessing
from tqdm import tqdm
import argparse
import torch

class GetBBox:
    def __init__(self,args):
        self.input_path=args["input"]
        self.save=args["save_image"] 
        self.img=None
        self.results=None
        self.crop=None
        self.center=None
        self.dimensions=None
        self.angle_of_rotation=None

    def get_bbox(self):
        self.img=cv2.imread(self.input_path)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        name=(self.input_path.split('\\')[-1]).split('.')[0] + ".png"
        self.results=model(self.input_path)
        xmin,ymin,xmax,ymax,_,_=self.results.xyxy[0][0]

        print(" Getting Crop")
        for i in tqdm([self.img]):
            self.crop=get_crop(i,xmin,ymin,xmax,ymax)
        
        self.center=get_center(xmin,ymin,xmax,ymax)

        print("Getting Contours")
        for c in tqdm([self.crop]):
            self.dimensions,self.angle_of_rotation = postprocessing(c)
        rect=(self.center,self.dimensions,self.angle_of_rotation)
        box=cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(self.img,[box],0,(0,0,255),2)
        cv2.imshow("test",self.img)
        if self.save:
            cv2.imwrite(name,self.img)
            print("image saved as ",name)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input",help="give path of the input image")
    parser.add_argument("-save","--save_image",nargs='?',const=True,default=False,help="save image given")
    args=vars(parser.parse_args())

    a=GetBBox(args)
    a.get_bbox()
