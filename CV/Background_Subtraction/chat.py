import matplotlib
matplotlib.use('gtk3agg')  # Use gtk3agg for GNOME

import skimage as ski
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os
import imageio
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

def extract_objects_from_binary(binary):
    # Label the connected regions in the binary image
    labels = label(binary)

    # Initialize an empty list to store each object
    objects = []

    # Loop through the detected regions and extract each one
    for region in regionprops(labels):
        if region.area >= 100:  # Only consider regions large enough (adjust as needed)
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

def main():
    # Read all images from the frames directory and convert them to grayscale
    image_list = []
    for filename in sorted(os.listdir('frames/')):
        image = imageio.v2.imread(f'frames/{filename}')
        gray_image = ski.color.rgb2gray(image)  # Convert to grayscale (0-1 float)
        image_list.append(gray_image)

    # Stack images into a video array
    video = np.array(image_list)

    # Calculate the background as the mean of all frames
    background_frame = np.mean(video, axis=0)  # axis=0 for frame-wise mean

    # Take the first frame and compute the difference from the background
    first_frame = video[0]
    difference = np.abs(first_frame - background_frame)

    # Apply Otsu's threshold to create a binary mask
    threshold = ski.filters.threshold_otsu(difference)
    binary = difference >= threshold  # Inverted threshold to detect foreground
    
    # Apply morphological closing to remove small holes and noise
    binary_cleaned = closing(binary, square(3))

    # Extract each object (white cluster) into a list
    objects = extract_objects_from_binary(binary_cleaned)

    # Set up the figure for plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(first_frame, cmap='gray')  # Show the first frame

    # Loop through each object and draw a bounding box
    for obj in objects:
        minr, minc, maxr, maxc = obj['bbox']  # Get the bounding box coordinates
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)  # Add the rectangle to the plot

    # Remove the axis and show the plot
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    # Optionally, save the output with bounding boxes as an image
    plt.savefig('overlay_with_boxes.png', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()

