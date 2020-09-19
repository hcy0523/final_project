#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  final.py
#  
#  Copyright 2020 Administrator <Administrator@HE-CHENGYANG>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  final.py
#  
#  Copyright 2020 Administrator <Administrator@HE-CHENGYANG>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


def main(args):
    return 0

if __name__ == '__main__':
	import os
	import cv2
	import sys
	import math
	import numpy as np
	from PIL import Image
	from PIL import ImageEnhance
	C=0
	figure_address='source_images'
	for root, dirs, files in os.walk(figure_address):
	  for d in dirs:
	    print(d) 
	for file in files:
		print(file)
		img_path = root+'/'+file
		img = cv2.imread(img_path,1)
		print(img_path, img.shape)
		
		#process
		C = C+1
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gauss = cv2.GaussianBlur(gray, (3, 3), 0)
		
		dst = np.zeros(gray.shape, np.uint8)
		cv2.normalize(gauss, dst, 255, 0,cv2.NORM_MINMAX, cv2.CV_8U)
		cv2.imwrite("process/processed"+str(C)+'.png',dst)
		##average and std
		(mean , stddv) = cv2.meanStdDev(dst)
		##thresholds
		Maxthr = mean + 3*stddv
		Minthr = mean - 3*stddv
		t1, lowthres = cv2.threshold(dst , Minthr, 255, cv2.THRESH_BINARY)
		
		t2, maxthres = cv2.threshold(dst, Maxthr, 255, cv2.THRESH_BINARY_INV)
		
		thres = cv2.bitwise_and(lowthres, maxthres)
		t3, thres = cv2.threshold(thres, 127, 255, cv2.THRESH_BINARY_INV)
		openkernel = np.ones((2,2), np.uint8)
		opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, openkernel)
		closekernel = np.ones((9,9), np.uint8)
		closing = cv2.morphologyEx(opening , cv2.MORPH_CLOSE, closekernel)
	
		openkernel = np.ones((3,3), np.uint8)
		opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, openkernel)
	
		erodekernel = np.ones((3,3), np.uint8)
		erode = cv2.morphologyEx(opening, cv2.MORPH_ERODE, erodekernel)

		contours,hierarchy =cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)	
		for i in range(len(contours)):
			length = cv2.arcLength(contours[i], True)
			area= cv2.contourArea(contours[i])
			if length < 25 or area < 15:
				cv2.drawContours(erode,[contours[i]],0,0,-1)		
			
		closekernel = np.ones((50,50), np.uint8)
		closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, closekernel)
	
		dilatekernel = np.ones((4,4), np.uint8)
		dilate = cv2.morphologyEx(closing , cv2.MORPH_DILATE, erodekernel)
	

		(mean , stddv) = cv2.meanStdDev(dilate)
		edges =cv2.Canny(dilate,mean/6,mean)
		cv2.imwrite("process/edges"+str(C)+'.png',edges)
		Con,hierarchy =cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = Con[:]
		
		# ~ dst_img = cv2.drawContours(img,Con,-1,(0,0,255),1)
		i = 0
		fname =open("information/"+str(C)+'.txt','w')
		fname.write(file+": "+'\n')
		if len(Con) == 0:	
			fname.write('No defect in this image\n')		
		else:		
			for i in range(len(Con)):
				x,y,w,h =cv2.boundingRect(Con[i])
				Con[i] = np.array([[[x,y]],[[x+w,y]], [[x+w,y+h]], [[x,y+h]]] )
				length = cv2.arcLength(contour[i], True)
				area = cv2.contourArea(contour[i])
				cnt=Con[i]
				right = tuple(cnt[cnt[:,:,0].argmax()][0])
				top = tuple(cnt[cnt[:,:,1].argmin()][0])
				bottom = tuple(cnt[cnt[:,:,1].argmax()][0])
				point=tuple([right[0],bottom[1]])
				cv2.putText(img,str(i+1), point, cv2.FONT_HERSHEY_COMPLEX,0.75,(0,255,255),2)
				fname.write('\n')
				fname.write('Defect'+'\t'+str(i+1)+'\n')
				fname.write('Length:'+'\t'+str(round(length))+'\n')
				fname.write('Area:'+'\t'+str(round(area))+'\n')
				fname.write(str(Con[i])+'\n')
			
		fname.close()			
		dst_img = cv2.drawContours(img,Con,-1,(0,0,255),2)
		cv2.imwrite("results/final"+str(C)+'.png',dst_img)
		cv2.waitKey(0)
	sys.exit(main(sys.argv))
