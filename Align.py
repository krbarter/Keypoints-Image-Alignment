import cv2
import rawpy
import numpy as np
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 10.24), dpi=96, facecolor='w', edgecolor='k')

# Class Input
from Dir import Directory

class Align:
    def __init__(self, img_input):
        print(img_input)
        first_path,second_path = img_input[0], img_input[1]
        self.shape = (960, 540)

        with rawpy.imread(first_path) as raw:
            rgb = raw.postprocess()
            imS = cv2.resize(rgb, self.shape)   # the image is blue
            self.first_image_colour = cv2.cvtColor(imS,cv2.COLOR_RGB2BGR)
            self.first_image = cv2.cvtColor(self.first_image_colour,cv2.COLOR_BGR2GRAY)
        
        with rawpy.imread(second_path) as raw2:
            rgb = raw2.postprocess()
            imS = cv2.resize(rgb, self.shape)
            self.second_image_colour =  cv2.cvtColor(imS,cv2.COLOR_RGB2BGR)
            self.second_image = cv2.cvtColor(self.second_image_colour,cv2.COLOR_BGR2GRAY)

    def getKeyPoints(self):
        # Getting ORD keypoints for both images
        orb = cv2.ORB_create()
        raw_points = orb.detect(self.first_image, None)
        self.first_image_with_points = cv2.drawKeypoints(self.first_image, raw_points, self.first_image, color=(0,255,0), flags=0)
        self.first_image_key_points = cv2.KeyPoint_convert(raw_points)
        print("Number of Keypoints, Image One: ", len(self.first_image_key_points))

        orb2 = cv2.ORB_create()
        raw_points = orb2.detect(self.second_image, None)
        self.second_image_with_points = cv2.drawKeypoints(self.second_image, raw_points, self.second_image, color=(0,255,0), flags=0)
        self.second_image_key_points = cv2.KeyPoint_convert(raw_points)
        print("Number of Keypoints, Image Two: ", len(self.second_image_key_points))

    def toints(self, points):
        for x in range(len(points)):
            for y in range(len(points[x])):
                points[x][y] = int(points[x][y])
        return points
    
    def keypointSorting(self, points):
        # Bubble sorting keypoints by X
        for i in range(len(points) - 1):
            for j in range(len(points) - i - 1):
                if points[j][0]> points[j + 1][0]:
                    points[j][0], points[j + 1][0] = points[j + 1][0], points[j][0]
        return points

    def getoffset(self, first_points, second_points):
        # Balanced By Length
        if len(first_points) < len(second_points):
            second_points = second_points[:len(first_points)]
        else:
            first_points = first_points[:len(second_points)]

        off_set_x = 0
        off_set_y = 0
        for x in range(len(first_points) - 1):
            off_set_x += (first_points[x][0] - second_points[x][0])
            off_set_y += (first_points[x][1] - second_points[x][1])

        off_set = [-off_set_x / 3000, -off_set_y / 3000]
        return off_set
    
    def imageAlignment(self):
        # Moving the first image along a Affine transform
        num_rows, num_cols = self.first_image.shape[:2]
        translation_matrix = np.float32([ [1,0,-self.off_set[1]], [0,1,-self.off_set[0]] ])
        img_translation = cv2.warpAffine(self.first_image_colour, translation_matrix, (num_cols, num_rows))
        return img_translation

    def keyPointEvaluation(self):
        # Evaluating by the sum of squares
        self.first_image_key_points_int  = Align.toints(self, self.first_image_key_points) 
        self.second_image_key_points_int = Align.toints(self, self.second_image_key_points)
        self.first_image_key_points_int_sorted  = Align.keypointSorting(self, self.first_image_key_points_int) 
        self.second_image_key_points_int_sorted = Align.keypointSorting(self, self.second_image_key_points_int) 

        self.off_set = Align.getoffset(self, self.first_image_key_points_int_sorted, self.second_image_key_points_int_sorted)
        aligned = Align.imageAlignment(self)
        combined_img_orriginal = cv2.addWeighted(aligned, 0.2, self.second_image_colour, 0.2, 0)
        
        print("Total Offset: ", self.off_set)
        cv2.imshow("Final", combined_img_orriginal)
        cv2.waitKey(0)

    def sceduler(self):
        Align.getKeyPoints(self)
        Align.keyPointEvaluation(self)

def __main__():
    #di = input("Directory Input: ")
    di = "Example/01L"
    a = Align(Directory(di).openDirectory())
    a.sceduler()
__main__()