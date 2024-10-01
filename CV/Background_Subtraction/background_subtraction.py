import matplotlib
matplotlib.use('gtk3agg')  # Use gtk3agg for GNOME
import skimage as ski
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import imageio
from skimage.measure import label, regionprops
from skimage.color import label2rgb

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
def draw_bounding_boxes(image,fg):
    labels = ski.measure.label(fg)
    regions = ski.measure.regionprops(labels)
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        rr,cc = ski.draw.rectangle_perimeter((minr,minc),(maxr,maxc),
            shape=image.shape)
        image[rr,cc] = 1
    return image

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
    
    plt.imshow(difference, cmap='gray')
    plt.axis('off')
    plt.show()
    # otsu method splits an image into two classes: foreground and background. Want to maximize the variance between classes. Split greyscale image into a binary image, so white is 1 and black is 0. Thats where the threshold comes into affect, values below threshold are 0, above 1. 

    threshold = ski.filters.threshold_otsu(difference) # image, bins, histogram
    binary = difference >= threshold #  example from link below
    plt.imshow(binary, cmap='binary')
    plt.axis('off')
    plt.show()
#  https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
        
    prof_list= [] 
    for filename in sorted(os.listdir('frames/')):
        current_frame = imageio.v2.imread(f'frames/{filename}')
        gray_image = ski.color.rgb2gray(current_frame)
        difference = np.abs(gray_image - background_frame)
        threshold = ski.filters.threshold_otsu(difference)
        binary = difference >= threshold
        frame = draw_bounding_boxes(current_frame, binary)
        prof_list.append(frame)

    prof_mask = np.array(prof_list)
    show_video(prof_mask, 'Using Professor Func to Draw Boxes')
def test():
    
    image_list = []
    for filename in sorted(os.listdir('frames/')):
        image = imageio.v2.imread(f'frames/{filename}')
        gray_image = ski.color.rgb2gray(image) #this turns each pixel into a float 0-1
        image_list.append(gray_image)
    video = np.array(image_list)

    background_frame = np.mean(video, axis=0)  # not sure why axis 0

    prof_list = []
    for filename in sorted(os.listdir('frames/')):
        current_frame = imageio.v2.imread(f'frames/{filename}')
        gray_image = ski.color.rgb2gray(current_frame)
        difference = np.abs(gray_image - background_frame)
        threshold = ski.filters.threshold_otsu(difference)
        binary = difference >= threshold
        frame = draw_bounding_boxes(current_frame, binary)
        prof_list.append(frame)

    prof_mask = np.array(prof_list)
    show_video(prof_mask, 'Using Professor Func to Draw Boxes')

if __name__ == "__main__":
    test()
