import cv2
import numpy as np

black = (0, 0, 0)

def overlay_mask(mask, image):
	#make the mask rgb
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #calculates the weightes sum of two arrays. in our case image arrays
    #input, how much to weight each. 
    #optional depth value set to 0 no need
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    # Copy
    image = image.copy()
    biggest_contour = 0
    
    _, contours, _= cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    if(contour_sizes):
        a=[True]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
        return a,biggest_contour, mask
    else:
        a=[False]
        return a,biggest_contour,image
    
def circle_contour(image, contour):
    # Bounding ellipse
    image_with_ellipse = image.copy()
    #easy function
    ellipse = cv2.fitEllipse(contour)
    #add it
    cv2.ellipse(image_with_ellipse, ellipse, black, 2, cv2.LINE_AA)
    return image_with_ellipse



def find_orange():
    #reading image and resizing it 
    image = cv2.imread("or.jpeg")
    max_dimension = max(image.shape)
        
    scale = 700/max_dimension
    
    image = cv2.resize(image, None, fx=scale, fy=scale)
    
    #appling filters and blur
    gb = cv2.GaussianBlur(image, (3,3), 1,1)
    
    dst = cv2.fastNlMeansDenoising(gb)
    
    #cnvt to hsv then masking the image
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv,(10, 100, 20), (22, 255, 255))
    
    #to increse brightness and intensity
    min_red2 = np.array([180, 100, 20])
    max_red2 = np.array([195, 255, 255])
    mask2 = cv2.inRange(hsv, min_red2, max_red2)
    
    mask = mask + mask2 
    
    #It is useful in closing small holes inside the foreground objects, 
    #or small black points on the object.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    #finding the biggest contour
    a,big_orange_contour, mask_orange = find_biggest_contour(mask_clean)
    
    if(any(a)):
        overlay = overlay_mask(mask_orange, image)
    
        #circling the image and making it a new image and returning it to ui
        result = circle_contour(overlay, big_orange_contour)
        
        name="output.jpeg"
        cv2.imshow(result)  
        cv2.waitKey(0);
''' else:
        return src'''
    
