import matplotlib
matplotlib.use('gtk3agg')  # Use gtk3agg for GNOME

import skimage as ski
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import imageio
import cv2
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
def extract_objects_from_binary(binary):
    # Label the connected regions in the binary image
    labels = label(binary)

    # Initialize an empty list to store each object
    objects = []

    # Loop through the detected regions and extract each one
    for region in regionprops(labels):
        if region.area >= 50:  # Only consider regions large enough (adjust as needed)
            # Create a mask for the region
            minr, minc, maxr, maxc = region.bbox
            object_mask = np.zeros(binary.shape, dtype=bool)
            object_mask[minr:maxr, minc:maxc] = (labels[minr:maxr, minc:maxc] == region.label)

            # Append the object's mask to the list
            objects.append({
                'mask': object_mask,
                'bbox': region.bbox  # Store the bounding box
            })

    return objects


def show_video(images,title):
    plt.title(title)
    for i,image in enumerate(images):
        if i == 0:
            obj = plt.imshow(image, cmap='gray')
            plt.axis('off') 
        else:
            obj.set_data(image)
        plt.pause(.01)
        plt.draw()
def display_video(video):

# Convert NumPy array to the right format for OpenCV (uint8)
    grayscale_frames = (video* 255).astype(np.uint8) #float -> uint8_t 

# Display the frames as a video using OpenCV
    for frame in grayscale_frames:
        cv2.imshow('Video', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):  # 50 millis sec per frame, 20 frames per second 
            break

    cv2.destroyAllWindows()
   
def boxes_prof(input_frame, binary_frame):

    labels = ski.measure.label(binary_frame)
    regions = ski.measure.regionprops(labels)
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-b', linewidth=2.5)

def draw_boxes(objects, first_frame):
    for obj in objects:
        # Extract the bounding box
        minr, minc, maxr, maxc = obj['bbox']

        first_frame[minr:minr+2, minc:maxc] = [0, 255, 0]  # Red top edge
        first_frame[maxr-2:maxr, minc:maxc] = [0, 255, 0]  # Red bottom edge
        
        # Draw the left and right edges of the rectangle
        first_frame[minr:maxr, minc:minc+2] = [0, 255, 0]  # Red left edge
        first_frame[minr:maxr, maxc-2:maxc] = [0, 255, 0]  # Red right edgefirst_frame[minr:minr+2, minc:maxc] = [0,255,0]

        # Create a Rectangle patch
    return first_frame 
def main():
    image_list = []
    for filename in sorted(os.listdir('frames/')):
        image = imageio.v2.imread(f'frames/{filename}')
        gray_image = ski.color.rgb2gray(image) #this turns each pixel into a float 0-1
        image_list.append(gray_image)
    video = np.array(image_list)
    show_video(video, 'prof img')
    #display_video(video)

    background_frame = np.mean(video, axis=0)  # not sure why axis 0
    plt.imshow(background_frame, cmap='gray')
    plt.axis('off')
    plt.show()
    
    """difference of first frame and background picture should just be the cars,
    in my mind because they are arrays, I can just do |a-b| i dont need 
    to index by index,
    there should be a built in func, which probably iterates
    """
    first_frame = imageio.v2.imread('frames/000000.jpg')  # np array
    # issue is background is np array is gray image
    first_frame_gray = ski.color.rgb2gray(first_frame)

    difference = np.abs(first_frame_gray - background_frame)
    print(difference.min())
    print(difference.max())
    
    plt.imshow(difference, cmap='gray')
    plt.axis('off')
    plt.show()
    # otsu method splits an image into two classes: foreground and background. Want to maximize the variance between classes. Split greyscale image into a binary image, so white is 1 and black is 0. Thats where the threshold comes into affect, values below threshold are 0, above 1. 

    threshold = ski.filters.threshold_otsu(difference) # image, bins, histogram
    print('threshold is ', threshold)
    binary = difference >= threshold #  example from link below
    print('binary is', binary)
    unique, counts = np.unique(binary, return_counts=True)
    print(f'Unique values in binary: {dict(zip(unique, counts))}')
    boxes_prof(None, binary)
    plt.imshow(binary, cmap='binary')
    plt.axis('off')
    plt.show()
#  https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
    
    objects = extract_objects_from_binary(binary)
    print(objects, first_frame)
    # Loop through each object and draw a bounding box
    draw_boxes(objects,first_frame) 

# remove artifacts connected to image border
    # Display the colored mask
    threshold_list= [] 
    for filename in sorted(os.listdir('frames/')):
        current_frame = imageio.v2.imread(f'frames/{filename}')
        gray_image = ski.color.rgb2gray(current_frame)
        difference = np.abs(gray_image - background_frame)
        #threshold = ski.filters.threshold_otsu(difference)
        binary = difference >= threshold
        objects = extract_objects_from_binary(binary)
        frame = draw_boxes(objects, current_frame)
        threshold_list.append(frame)

    video_mask = np.array(threshold_list)
    show_video(video_mask, 'binary masks')
if __name__ == "__main__":
    main()
