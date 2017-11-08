#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

#reading in an image
image = mpimg.imread('../test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    #One approach is to find the farthest points in each side and draw a line between them. that is what
    #the function below does.
    #this approach however leaves an initial gap when dashed lane lines are encountered
    
    #draw_farthest(img, lines, color=[0, 0, 255], thickness=2)
    
        
    #PolyFit approach: 
    draw_polyline(img, lines, color=[0, 0, 255], thickness=5)
        
#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def seperate_lines_by_slope(lines):
    """
    Seperate the given line segments into groups based on their slope
    Positive slope is associated with the left marking and the negative with the right.
    """
    left_group=[]
    right_group=[]
        
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1!=x2:
                if (y2-y1)/(x2-x1) > 0:
                    left_group.append([x1,y1])
                    left_group.append([x2,y2])
                else:
                    right_group.append([x1,y1])
                    right_group.append([x2,y2])
            else:
                continue
    return np.array(left_group), np.array(right_group)

def seperate_lines_polyfit(lines):
    """
    Seperate the given lines based on proximity to the left and right edges of the image,
    compared to the polyfit of all points (solving the problem of infinite slope.)
    """
    
# First get a fit between the lines to use as a reference, for deciding the lane side

    all_points = get_points_from_lines(lines)
    
    median_coefs = np.polyfit(all_points[:,0].ravel(), all_points[:,1].ravel(), 1)
    
    left_group=[]
    right_group=[]
        
    for x,y in all_points:
        if x < x_from_y(y, median_coefs):
            left_group.append([x,y])
        else:
            right_group.append([x,y])
            
    return np.array(left_group), np.array(right_group)

def seperate_lines(lines, imshape):
    all_points = get_points_from_lines(lines)
    
    left_group=[]
    right_group=[]
	
    for x,y in all_points:
        if x < imshape[1]/2 :
            left_group.append([x,y])
        else:
            right_group.append([x,y])
           
    return np.array(left_group), np.array(right_group)

def x_from_y(y,coefs):
    
    return int((y-coefs[1])/coefs[0])

def distance_between(x1,y1,x2,y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def get_points_from_lines(lines):
    
    points = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            points.append([x1,y1])
            points.append([x2,y2])
    return np.array(points)
    
def farthest_indices(points):
    
    from scipy.spatial.distance import pdist, squareform
    dist = squareform(pdist(points))
    
    return np.unravel_index(np.argmax(dist), dist.shape)
    
def draw_farthest(img, lines, color=[0, 0, 255], thickness=2):
    m1_group, m2_group = seperate_lines(lines)

    left_p = np.array(m1_group)
    right_p = np.array(m2_group)
    
    lx1y1_index, lx2y2_index = farthest_indices(left_p)
    rx1y1_index, rx2y2_index = farthest_indices(right_p)
 
    cv2.line(img, tuple(left_p[lx1y1_index]), tuple(left_p[lx2y2_index]), color, thickness)
    cv2.line(img, tuple(right_p[rx1y1_index]), tuple(right_p[rx2y2_index]), color, thickness)
    
left_coefs = []
right_coefs = []	
	
def draw_polyline(img, lines, color=[0, 0, 255], thickness=5):
    left_points, right_points = seperate_lines(lines, img.shape)

#    print("left points shape:  ", left_points.shape)
#    print("right points shape: ", right_points.shape)
    
    global left_coefs
    global right_coefs
	
    if left_points.size!=0: left_coefs = np.polyfit(left_points[:,:1].ravel(), left_points[:,1:].ravel(), 1)
    if right_points.size!=0: right_coefs = np.polyfit(right_points[:,:1].ravel(), right_points[:,1:].ravel(), 1)
    
    #get start and end points using y=mx+b
    
    left_start_y = img.shape[0]
    right_start_y = img.shape[0]
    
    left_start_x = int((left_start_y - left_coefs[1])/left_coefs[0])
    right_start_x = int((right_start_y - right_coefs[1])/right_coefs[0])


    left_end_y = int(img.shape[0]/2 + 50)
    right_end_y = int(img.shape[0]/2 + 50)
    
    left_end_x = int((left_end_y - left_coefs[1])/left_coefs[0])
    right_end_x = int((right_end_y - right_coefs[1])/right_coefs[0])
    
    cv2.line(img, (left_start_x, left_start_y), (left_end_x, left_end_y), color, thickness)
    cv2.line(img, (right_start_x, right_start_y), (right_end_x, right_end_y), color, thickness)
	
def process_dir(dirin, dirout):	
	import os
	dir_list = os.listdir(dirin)

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

	for filename in dir_list:
		image = cv2.imread(dirin + '/' + (filename))
		print(filename + "\n")
		gray = grayscale(image)

#Apply Gaussian
		blur_gray = gaussian_blur(gray,5)

#Detect edges using Canny
		edges = canny(blur_gray, 50,110)
		cv2.imwrite(dirout + '/canny/'+filename, edges)

#Create and apply region mask
		imshape = edges.shape
		verts = np.array([[(imshape[1]/2-25,imshape[0]/2+50), (0,imshape[0]), (imshape[1],imshape[0]), (imshape[1]/2+25,imshape[0]/2+50)]], dtype=np.int32)
		masked = region_of_interest(edges,verts)

#Find lines using Hough transform
		img_lines = hough_lines(masked, 1, np.pi/180, 15, 40, 15)

#overlaying on original
		combined = weighted_img(img_lines, image)
		cv2.imwrite(dirout + '/'+filename, combined)

def process_img(img_file):
	image = cv2.imread(img_file)
	gray = grayscale(image)

#Apply Gaussian
	blur_gray = gaussian_blur(gray,5)

#Detect edges using Canny
	edges = canny(blur_gray, 50,110)

#Create and apply region mask
	imshape = edges.shape
	verts = np.array([[(imshape[1]/2-25,imshape[0]/2+50), (0,imshape[0]), (imshape[1],imshape[0]), (imshape[1]/2+25,imshape[0]/2+50)]], dtype=np.int32)
	masked = region_of_interest(edges,verts)

#Find lines using Hough transform
	img_lines = hough_lines(masked, 1, np.pi/180, 15, 40, 15)

#overlaying on original
	combined = weighted_img(img_lines, image)
	cv2.imshow('img_combined', combined)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
		
#process_img("test_images/310.jpg")		

process_dir("test_images", "test_images_output")
