#-*- coding: UTF-8 -*-  
import cv2
import numpy as np
import math
from scipy import signal
from PIL import Image
from PIL import ImageEnhance


def powerTrans(gray, c, y):
    gray_power = np.uint8(c * (gray ** y))
    return gray_power
    
def plf(x, X, Y):
    x = np.float32(x)
    y = 0
    for i in range(len(X)):
        t = 1
        for j in range(len(Y)):
            if j != i:
                t = t * ((x - X[j]) / (X[i] - X[j]))
        y = np.uint8(y + t * Y[i])
    return y

def fft2Image(src):
    r,c = src.shape[:2]
    rPadded = cv2.getOptimalDFTSize(r)
    cPadded = cv2.getOptimalDFTSize(c)
    fft2 = np.zeros((rPadded,cPadded, 2),np.float32)
    fft2[:r,:c, 0] =src
    cv2.dft(fft2,fft2,cv2.DFT_COMPLEX_OUTPUT)
    return fft2

def amplitudeSpectrum(fft2):
    real2 = np.power(fft2[:,:,0],2.0)
    Imag2 = np.power(fft2[:,:,1],2.0)
    amplitude = np.sqrt(real2+Imag2)
    return amplitude
    
def graySpectrum(amplitude):
	amplitude = np.log(amplitude +1.0)
	spectrum = np.zeros(amplitude.shape, np. float32)
	cv2.normalize(amplitude, spectrum, 0, 1, cv2.NORM_MINMAX)
	return spectrum

def grayReversal(gray):
    gray_reversal = 255 - gray 
    return gray_reversal
    
def getGaussKernel(sigma,H,W):

	gaussMatrix = np.zeros([H,W], np.float32)
	cH = (H-1)/2
	cW = (W-1)/2
	
	for r in range(H):
		for c in range(W):
			norm2 = math.pow(r-cH,2) + math.pow(c-cW,2)
			gaussMatrix[r][c] =math.exp(-norm2/(2*math.pow(sigma,2)))
		
	sumGM = np.sum(gaussMatrix)
	gaussKernel = gaussMatrix/sumGM
	
	return gaussKernel
	
def gaussBlur(image, sigma, H, W, _boundary ='fill', _fillvalue = 0):
    
	gaussKernel_x = cv2.getGaussianKernel(sigma, W, cv2.CV_64F)
	gaussKernel_x = np.transpose(gaussKernel_x)
	gaussBlur_x = signal.convolve2d(image, gaussKernel_x, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
	
	gaussKernel_y = cv2.getGaussianKernel(sigma, H, cv2.CV_64F)
	gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
	
	return gaussBlur_xy

def calcGrayHist(image):
    
    rows,cols = image.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):	
        for c in range(cols):
            grayHist[image[r][c]] +=1
            
    return grayHist  

def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def gaussian_noise(image):          
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]   # blue
            g = image[row, col, 1]   # green
            r = image[row, col, 2]   # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv2.imshow("noise image", image)


def otsu(image):
    
    rows,cols = image.shape
    grayHist = calcGrayHist(image)
    uniformGrayHist = grayHist/float(rows*cols)
    zeroCumuMoment = np.zeros([256], np.float32)
    oneCumuMoment = np.zeros([256], np.float32)

    for k in range(256):
	
	    if k == 0:		
		    zeroCumuMoment[k] = uniformGrayHist[0]
		    oneCumuMoment[k] = (k)*uniformGrayHist[0]
		
	    else:
		    zeroCumuMoment[k] = zeroCumuMoment[k-1] + uniformGrayHist[k]
		    oneCumuMoment[k] = oneCumuMoment[k-1] + k*uniformGrayHist[k]

    variance = np.zeros([256], np.float32)
    
    for k in range(255):
	
	    if zeroCumuMoment[k] == 0 or zeroCumuMoment[k] == 1:
		    variance[k] = 0
		   
	    else:
		    variance[k] = math.pow(oneCumuMoment[255]*zeroCumuMoment[k] - oneCumuMoment[k],2)/(zeroCumuMoment[k]*(1.0 - zeroCumuMoment[k]))
	
    threshLoc = np.where(variance[0:255] == np.max(variance[0:255]))
    thresh = threshLoc[0][0]
    threshold = np.copy(image)
    threshold[threshold > thresh] = 255
    threshold[threshold <= thresh] = 0
    
    return (thresh,threshold)

def createLPFilter(shape, center, radius, lpType, n=2):

    rows, cols =shape[:2]
    r,c = np.mgrid[0:rows:1, 0:cols:1]
    c-= center[0]
    r-= center[1]
    d = np.power(c,2.0) + np.power(r,2.0)
    lpFilter = np.zeros(shape, np.float32)
    
    if (radius<=0):
	    return lpFilter
	    
    if (lpType == 0):
        lpFilter = np.copy(d)
        lpFilter[lpFilter<pow(radius,2.0)] = 1
        lpFilter[lpFilter<=pow(radius,2.0)] = 0
    elif (lpType == 1):
        lpFilter = 1.0/(1.0+np.power(np.sqrt(d)/radius,2*n))
	    
    elif (lpType == 2):
        lpFilter = np.exp(-d/(2.0*pow(radius,2.0)))
        
    return lpFilter
    
def seed_fill(img):
	ret,img = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
	label = 100
	stack_list = []
	h,w = img.shape
	for i in range(1,h-1,1):
		for j in range(1,w-1,1):
			if (img[i][j] == 255):
				img[i][j] = label
				stack_list.append((i,j))
				while len(stack_list)!=0:
					cur_i = stack_list[-1][0]
					cur_j = stack_list[-1][1]
					img[cur_i][cur_j] = label
					stack_list.remove(stack_list[-1])
					
					if (img[cur_i-1][cur_j-1] == 255):
						stack_list.append((cur_i-1,cur_j-1))
					if (img[cur_i-1][cur_j] == 255):
						stack_list.append((cur_i-1,cur_j))
					if (img[cur_i-1][cur_j+1] == 255):
						stack_list.append((cur_i-1,cur_j+1))
					if (img[cur_i][cur_j-1] == 255):
						stack_list.append((cur_i,cur_j-1))
					if (img[cur_i+1][cur_j-1] == 255):
						stack_list.append((cur_i+1,cur_j-1))
					if (img[cur_i+1][cur_j] == 255):
						stack_list.append((cur_i+1,cur_j))
					if (img[cur_i][cur_j+1] == 255):
						stack_list.append((cur_i,cur_j+1))
					if (img[cur_i+1][cur_j+1] == 255):
						stack_list.append((cur_i+1,cur_j+1))

def originalSeed(gray,thresh):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))#3×3结构元
	
	thresh_copy = thresh.copy() #复制thresh_A到thresh_copy
	thresh_B = np.zeros(gray.shape, np.uint8) #thresh_B大小与A相同，像素值为0
	seeds = [ ] #为了记录种子坐标
	
	#循环，直到thresh_copy中的像素值全部为0
	while thresh_copy.any():
	
		Xa_copy, Ya_copy = np.where(thresh_copy > 0) #thresh_A_copy中值为255的像素的坐标
		thresh_B[Xa_copy[0], Ya_copy[0]] = 255 #选取第一个点，并将thresh_B中对应像素值改为255
	
		#连通分量算法，先对thresh_B进行膨胀，再和thresh执行and操作（取交集）
		for i in range(200):
			dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
			thresh_B = cv2.bitwise_and(thresh, dilation_B)
	
		#取thresh_B值为255的像素坐标，并将thresh_copy中对应坐标像素值变为0
		Xb, Yb = np.where(thresh_B > 0)
		thresh_copy[Xb, Yb] = 0
	
		#循环，在thresh_B中只有一个像素点时停止
		while str(thresh_B.tolist()).count("255") > 1:
			thresh_B = cv2.erode(thresh_B,  kernel, iterations=1) #腐蚀操作
	
		X_seed, Y_seed = np.where(thresh_B > 0) #取处种子坐标
		if X_seed.size > 0 and Y_seed.size > 0:
			seeds.append((X_seed[0], Y_seed[0]))#将种子坐标写入seeds
		thresh_B[Xb, Yb] = 0 #将thresh_B像素值置零
	
	return seeds

#区域生长
def regionGrow(gray, seeds, thresh, p):
    seedMark = np.zeros(gray.shape)
    #八邻域
    if p == 8:
        connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    elif p == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    #seeds内无元素时候生长停止
    while len(seeds) != 0:
        #栈顶元素出栈
        pt = seeds.pop(0)
        for i in range(p):
            tmpX = pt[0] + connection[i][0]
            tmpY = pt[1] + connection[i][1]

            #检测边界点
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue

            if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = 255
                seeds.append((tmpX, tmpY))
    return seedMark
	
