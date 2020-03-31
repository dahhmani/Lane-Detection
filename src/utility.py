import numpy as np
import cv2 as cv

index = {'challenge': 0, 'project': 1}
leftFit_prev, rightFit_prev = [], []

def laneFit(binaryFrame, polyDegree):
    global leftFit_prev
    global rightFit_prev

    #Hyper-parameters
    no_windows = 9 # number of sliding windows
    windowWidth = 100 # +/- margin
    minPixels = 50 # minimum number of pixels found to recenter window
    
    processedFrame = np.dstack((binaryFrame, binaryFrame, binaryFrame))

    histogram = np.sum(binaryFrame[binaryFrame.shape[0]//2:,:], axis=0) # only for the bottom half of the image
    windowHeight = np.int(binaryFrame.shape[0]/no_windows)

    midpoint = np.int(histogram.size/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = binaryFrame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftLanePixels, rightLanePixels = [], []

    for window in range(no_windows):
        # Define the windows
        winy_low = binaryFrame.shape[0] - (window+1)*windowHeight
        winy_high = binaryFrame.shape[0] - window*windowHeight
        leftwinx_low = leftx_base - windowWidth
        leftwinx_high = leftx_base + windowWidth
        rightwinx_low = rightx_base - windowWidth
        rightwinx_high = rightx_base + windowWidth
        # Draw the windows
        cv.rectangle(processedFrame,(leftwinx_low,winy_low),(leftwinx_high,winy_high),(0,255,0), 2)
        cv.rectangle(processedFrame,(rightwinx_low,winy_low),(rightwinx_high,winy_high),(0,255,0), 2)
        # Find nonzero pixels in x and y inside the windows
        leftPixels = ((nonzeroy >= winy_low) & (nonzeroy < winy_high) & (nonzerox >= leftwinx_low) & (nonzerox < leftwinx_high)).nonzero()[0]
        rightPixels = ((nonzeroy >= winy_low) & (nonzeroy < winy_high) & (nonzerox >= rightwinx_low) & (nonzerox < rightwinx_high)).nonzero()[0]
        leftLanePixels.append(leftPixels)
        rightLanePixels.append(rightPixels)
        # Recenter next windows if necessary
        if len(leftPixels) > minPixels:
            leftx_base = np.int(np.mean(nonzerox[leftPixels]))
        if len(rightPixels) > minPixels:
            rightx_base = np.int(np.mean(nonzerox[rightPixels]))

    leftLanePixels = np.concatenate(leftLanePixels)
    rightLanePixels = np.concatenate(rightLanePixels)

    leftx = nonzerox[leftLanePixels]
    lefty = nonzeroy[leftLanePixels]
    rightx = nonzerox[rightLanePixels]
    righty = nonzeroy[rightLanePixels]

    if lefty.size == 0:
        leftFit = leftFit_prev
    else:
        leftFit = np.polyfit(lefty, leftx, polyDegree)
        leftFit_prev = leftFit

    if righty.size == 0:
        rightFit = rightFit_prev
    else:
        rightFit = np.polyfit(righty, rightx, polyDegree)
        rightFit_prev = rightFit
    
    processedFrame[nonzeroy[leftLanePixels], nonzerox[leftLanePixels]] = [255, 0, 0]
    processedFrame[nonzeroy[rightLanePixels], nonzerox[rightLanePixels]] = [0, 0, 255]

    return leftFit, rightFit, processedFrame

def binarize(frame, K, dist, src, dst, i):
    undistorted = cv.undistort(frame, K, dist)    
    warped = warp(undistorted, src, dst, undo=False)
       
    if i == index['challenge']: 
        yellow_lower = np.array([0,120,51])
        yellow_upper = np.array([33,157,255])
        white_lower = np.array([0,0,205])
        white_upper = np.array([255,255,255])

        hls = cv.cvtColor(warped, cv.COLOR_BGR2HLS)
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV) 
        yellow_mask = cv.inRange(hls, yellow_lower, yellow_upper)
        white_mask = cv.inRange(hsv, white_lower, white_upper)
        binaryFrame = cv.bitwise_or(yellow_mask, white_mask)

    elif i == index['project']:
        whiteLower = np.array([0, 0, 211]) 
        whiteUpper = np.array([255, 11, 255])
        
        hls = cv.cvtColor(warped, cv.COLOR_BGR2HLS)
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        white_mask = cv.inRange(hls, whiteLower, whiteUpper)
        frame = cv.bitwise_or(gray, white_mask)
        thresh = 159
        binaryFrame = cv.threshold(frame, thresh, 255, cv.THRESH_BINARY)[1]

    return binaryFrame, warped

def warp(frame, src, dst, undo=False):        
    if undo:
        src, dst = dst, src

    frameSize = (frame.shape[1], frame.shape[0])
    H = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(frame, H, frameSize) # keep same size as input image

    return warped

def constructRegressionMatrix(x_dataset, degree):
    X_dataset = np.ones((len(x_dataset),1))
    for i in range(1,degree+1):
        X_dataset = np.column_stack((x_dataset**i, X_dataset))

    return X_dataset

def videoParameters(i):
    ''' K (Camera Matrix), dist (Distortion Coefficients), src & dst (Warp Points) '''

    if i == index['challenge']:
        K = np.array([
            [1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
            [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ], dtype=np.float32)
        dist = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02], dtype=np.float32)
        src = np.array([
            [190, 700], 
            [1110, 700], 
            [720, 470], 
            [570, 470],
        ], dtype=np.float32)
        dst = np.array([
            [250, 720],
            [1035, 720],
            [1035, 0],
            [250, 0],
        ], dtype=np.float32)

    elif i == index['project']:
        K = np.array([
            [9.037596e+02, 0.000000e+00, 6.957519e+02], 
            [0.000000e+00, 9.019653e+02, 2.242509e+02], 
            [0.000000e+00, 0.000000e+00, 1.000000e+00]
        ], dtype = np.float32)
        dist = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]], dtype = np.float32)
        src = np.array([
            [203, 510],
            [940, 510],
            [745, 300],
            [545, 300],
        ], dtype=np.float32)
        dst = np.array([
            [250, 510],
            [1140, 510],
            [1140, 0],
            [250, 0],
        ], dtype=np.float32)

    return (K, dist, src, dst)

def updateTrackbars(xmax, ymax):
    yBottom = cv.getTrackbarPos('yBottom', 'Control Points') / 1000*ymax
    xBottom_L = cv.getTrackbarPos('xBottom_L', 'Control Points') / 1000*xmax
    xBottom_R = cv.getTrackbarPos('xBottom_R', 'Control Points') / 1000*xmax
    yTop = cv.getTrackbarPos('yTop', 'Control Points') / 1000*ymax
    xTop_L = cv.getTrackbarPos('xTop_L', 'Control Points') / 1000*xmax
    xTop_R = cv.getTrackbarPos('xTop_R', 'Control Points') / 1000*xmax

    src = np.array([
            [xBottom_L, yBottom],
            [xBottom_R, yBottom], 
            [xTop_R, yTop], 
            [xTop_L, yTop], 
        ], dtype=np.float32)

    return src

def initializeTrackbars(xmax, ymax, src):
    def nothing(arg): pass

    yBottom, xBottom_L, xBottom_R, yTop, xTop_R, xTop_L = src[0][1], src[0][0], src[1][0], src[2][1], src[2][0], src[3][0] 
    cv.namedWindow('Control Points')
    cv.resizeWindow('Control Points', 360, 240)
    cv.createTrackbar('yBottom', 'Control Points', int(round(yBottom/ymax*1000)), 1000, nothing)
    cv.createTrackbar('xBottom_L', 'Control Points', int(round(xBottom_L/xmax*1000)), 1000, nothing)
    cv.createTrackbar('xBottom_R', 'Control Points', int(round(xBottom_R/xmax*1000)), 1000, nothing)
    cv.createTrackbar('yTop', 'Control Points', int(round(yTop/ymax*1000)), 1000, nothing)
    cv.createTrackbar('xTop_L', 'Control Points', int(round(xTop_L/xmax*1000)), 1000, nothing)
    cv.createTrackbar('xTop_R', 'Control Points', int(round(xTop_R/xmax*1000)), 1000, nothing)

def imshow(windowName, image):
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.imshow(windowName, image)

def drawPoints(frame, points):
    pts = np.int32(np.around(points))
    for i in range(pts.shape[0]):
        cv.circle(frame, (pts[i][0],pts[i][1]), 15, (0,0,255), cv.FILLED)
    cv.drawContours(frame, [pts], -1, (0,0,255), 2)

def visualizePipeline(inputFrame, warped, binaryWarped, processed, src, dst):
    drawPoints(inputFrame, src)
    drawPoints(warped, dst)
    pipeline = stackImages([[inputFrame, warped], [binaryWarped, processed]], [['original', 'warped'], ['binaryWarped', 'processed']])
    imshow('Pipeline', pipeline)

def stackImages(imgArray, lables=[], scale=1):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range (0,cols):
                cv.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv.FILLED)
                cv.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    
    return ver