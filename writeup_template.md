# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

There are two parts to this problem: Information extraction and Processing it to draw the needed lines

A. Information Extraction:
1. This part of the pipeline converts the image to grayscale, so that it can be passed to the Canny edge detection function, to extract only the edges from the image, after it is blurred to reduce noise.
2. The output of the above step is fed to a region selection function, that removes all pixels except the ones in the narrowing region of interest where we expect to find the lane lines ahead of the car.
3. Once the edges are masked to only the region of interest, Hough transform technique is used to detect lines from the edge pixels, which is fed to the second part of the pipeline - data processing

B. Processing:
1. The lines (point pairs) returned by the HoughPLines function need to be seperated between the ones belonging to the left lane and the right lane. Several different approaches were tried: 
    1.1. Slopes: Left lane lines would have a positive slope and the right ones, negative. Problem with this approach is when the camera is aligned too close to a line, the slope becomes infinite and does not help with the seperation.
    1.2 Polyfit: thi approach takes all the points and tries to fit a line between all points. This didn't work well due to the nature of how polfit works.
    1.2 left to left: Any points that lie on the left half of the image are assigned to the left lane and likewise for the right.
2. Once the points are seperated in the two groups, they could be extrapolated using either of the two approaches:
    2.1 Farthes points: Here the distance between all  possible combinations between points was calculated to determine the farthest points. Once known, these points were then used to calculate slope and intercept of the average lane lines, followed by the drawing it out on the image with the height of the image as the starting y value, until the last point found earlier.
    2.2 Polyfit: np.polyfit was carried out to determine the best fit slope and intercept of left and right point groups, followed by drawing the lines starting at the bottom of the image to the center.

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
