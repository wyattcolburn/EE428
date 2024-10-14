import glob
import os
import imageio
import skimage as ski
import numpy as np
import matplotlib
matplotlib.use('gtk3agg')  # Use gtk3agg for GNOME
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from skimage.feature import SIFT, match_descriptors
from sklearn.cluster import KMeans

"""
Resources used:
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.SIFT
"""
def grey_list(paths):
    
    gray_list =[] #for path in paths:
    detector_list = []
    i = 0
    for path in paths:    
        image = imageio.v2.imread(path)
        gray_image = ski.color.rgb2gray(image) #this turns each pixel into a float 0-1
        gray_list.append(gray_image)
        sift_stuff(gray_image, detector_list)
        print(i)
        i+=1
    return sift_list

def sift_stuff(image, list):
    #descriptors are the local image gradient info, around the keypoint
    descriptor_extractor= SIFT()
    print("sifting")
    descriptor_extractor.detect_and_extract(image)
    list.append(descriptor_extractor.descriptors)

def sift_average(list):

    kmeans = sklean.cluster.KMeans()
    mean_list = kmeans.fit(list)
    cluster_list = []
    for item, i  in enumerate(list):
        val = kmeans.predict(list[i])
        cluster_list.append(val)
    
    return kmeans

def main():

    paths = sorted(glob.glob('Childrens-Books/*.jpg'))
    gray_list = grey_list(paths)

if __name__ == "__main__":
    main()
