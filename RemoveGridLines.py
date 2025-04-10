import numpy as np
import cv2

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

def remove_grid_lines(filePath):

    src = cv2.imread(filePath, cv2.IMREAD_COLOR)
    
    if src is None:
        print ('Error opening image: ' + filePath)
        return -1
    
    show_wait_destroy('src', src)

    # Transform source image to gray if it isn't already
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    show_wait_destroy('gray', gray)
    
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 15, -2)
    show_wait_destroy('binary', bw)

    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = cols // 45

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Here we shrink everything down until only the horizontal lines remain
    # and then we dilate a bunch to get the lines back to proper thickness
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations = 5)
    rebuildLineKernel = np.ones((1, horizontal_size), np.uint8)
    horizontal = cv2.dilate(horizontal, rebuildLineKernel, iterations = 50)

    show_wait_destroy('horizontal', horizontal)

    rows = vertical.shape[0]
    verticalsize = rows // 45

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    vertical = cv2.erode(vertical, verticalStructure, iterations = 5)
    rebuildLineKernel = np.ones((verticalsize, 1), np.uint8)
    vertical = cv2.dilate(vertical, rebuildLineKernel, iterations = 50)
    show_wait_destroy('vertical', vertical)

    vertical = cv2.bitwise_not(vertical)
    show_wait_destroy('vertical_bit', vertical)


    # Next have to subtract the horizontal and vertical from the original
    
    horiKernel = np.ones((3, 1), np.uint8)
    horizontal = cv2.bitwise_not(horizontal)
    horizontal = cv2.erode(horizontal, horiKernel)
    show_wait_destroy('hori inverse', horizontal)

    combined = vertical
    (rows, cols) = np.where(horizontal == 0)
    combined[rows, cols] = 0

    show_wait_destroy('combined', combined)

    (outputRows, outputCols) = np.where(combined == 0)
    output = bw
    output[outputRows, outputCols] = 0
    kernel = np.array([[1, 1, 1],
                       [0, 1, 0],
                       [1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], np.uint8)

    output = cv2.dilate(output, kernel)
    kernel = np.ones((2,2), np.uint8)
    output = cv2.erode(output, kernel)
    output = cv2.bitwise_not(output)
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, horiKernel, iterations = 1)
    show_wait_destroy('output', output)

    return output

remove_grid_lines("image.png")