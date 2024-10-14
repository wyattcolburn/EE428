import glob
import os
import imageio
import skimage as ski
import numpy as np
import matplotlib
matplotlib.use('gtk3agg')  # Use gtk3agg for GNOME
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

def main():

    paths = sorted(glob.glob('Childrens-Books/*.jpg'))
    print(paths) 
    gray_list =[]
    for path in paths:
        
        image = imageio.v2.imread(path)
        gray_image = ski.color.rgb2gray(image) #this turns each pixel into a float 0-1
        gray_list.append(gray_image)

    print(gray_list)


if __name__ == "__main__":
    main()
