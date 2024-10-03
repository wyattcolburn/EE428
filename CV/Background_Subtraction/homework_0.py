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
    
    first_frame = imageio.v2.imread('frames/000000.jpg')  # np array
    # issue is background is np array is gray image
    first_frame_gray = ski.color.rgb2gray(first_frame)

    difference = np.abs(first_frame_gray - background_frame)
    
    plt.imshow(difference, cmap='gray')
    plt.axis('off')
    plt.show()
    
    threshold = ski.filters.threshold_otsu(difference) # image, bins, histogram
    binary = difference >= threshold #  example from link below, values not in the background become a 1
    plt.imshow(binary, cmap='binary')
    plt.axis('off')
    plt.show()
#  https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
        
    output_list= [] 
    for filename in sorted(os.listdir('frames/')):
        current_frame = imageio.v2.imread(f'frames/{filename}')
        gray_image = ski.color.rgb2gray(current_frame)
        difference = np.abs(gray_image - background_frame)
        threshold = ski.filters.threshold_otsu(difference)
        binary = difference >= threshold
        frame = draw_bounding_boxes(current_frame, binary)
        output_list.append(frame)

    output_video = np.array(output_list)
    show_video(output_video, 'Output Video')

if __name__ == "__main__":
   main()
