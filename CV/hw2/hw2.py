import glob
import os
import imageio
import skimage as ski
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from skimage.feature import SIFT, match_descriptors
from sklearn.cluster import KMeans
import pickle
import argparse


from sklearn.feature_extraction.text import TfidfVectorizer
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
from sklearn.neighbors import NearestNeighbors
#https://scikit-learn.org/0.15/modules/generated/sklearn.neighbors.NearestNeighbors.html
"""
Resources used:
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.SIFT
"""
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
    return detector_list

def sift_stuff(image, list):
    #descriptors are the local image gradient info, around the keypoint
    descriptor_extractor= SIFT()
    descriptor_extractor.detect_and_extract(image)
    list.append(descriptor_extractor.descriptors) # appending x amount of descriptors, each descriptor has 128 features

def display_neighbors(indices, paths, path_query, query_number):
    fig, axes = plt.subplots(1, 6, figsize=(15, 6))  # Create 5 subplots horizontally
    query_image = imageio.v2.imread(path_query[query_number])
    plt.imshow(query_image)
    axes[0].imshow(query_image)
    axes[0].axis("off")
# Loop through the paths and display each image in a subplot
    for i, ax in enumerate(axes[1:],start=1 ):
        image_index = indices[i-1]
        image = imageio.v2.imread(paths[image_index])  # Read the image from path
        ax.imshow(image)                  # Show the image
        ax.axis('off')                    # Hide axes for each subplot

    plt.show()

def kmeans_training(sift_list):

    kmeans = KMeans(n_clusters = 1000)
    flatten_sift_list =  np.vstack(sift_list)
    mean_list = kmeans.fit(flatten_sift_list)
    return kmeans
def predict(input_list, kmeans):
    cluster_labels = []
    for input in input_list:
        if input is not None and len(input) > 0:
            val = kmeans.predict(input)
            cluster_labels.append(val)
        else:
            print("input not valid")
    return cluster_labels 

def text_document(cluster_labels):
    document_list = []
    for index, cluster in enumerate(cluster_labels):
        local_string=""
        for x in range(len(cluster)):
            local_string +=str(cluster[x]) + " "
        document_list.append(local_string)

    return document_list

def input(path, kmeans, vectorizer, knn, query_index):
    query_sift = grey_list(path)
    cluster_labels = predict(query_sift, kmeans)
    document_list = text_document(cluster_labels)

    query_transform = vectorizer.transform([document_list[query_index]])
    distances, indices = knn.kneighbors(query_transform)
    indices = np.array(indices)
    
    indices = indices.flatten().tolist()
    return indices

def load_sift_list():

    with open("data.pkl", "rb") as f:
        sift_list = pickle.load(f)
    return sift_list


def load_kmeans():
    with open("kmeans_data.pkl", "rb") as f:
        kmeans = pickle.load(f)

    return kmeans

def filename_index(filename, path_query, path_extra):
    
    if filename not in path_query and filename not in path_extra:
        print("invalid filename")
        exit()

    elif filename in path_query: 
        index = path_query.index(filename)
        return index, path_query
    else:
        print("conditional block exitted")
        index = path_extra.index(filename)
        return index, path_extra
def main():
    

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    print(args.filename)
    paths = sorted(glob.glob('Childrens-Books/*.jpg'))
    path_query = sorted(glob.glob('queries/*.jpg'))
    path_extra = sorted(glob.glob('extra-queries/*.jpg'))
    
    input_index, input_path = filename_index(args.filename, path_query, path_extra)
    
    #sift_list = grey_list(paths) how i would get the sift_list if I didn't store it to a pickle file
    sift_list = load_sift_list()
    kmeans = load_kmeans()
    #kmeans = kmeans_training(sift_list) same thing, how it was created before loadign it 

    cluster_labels = predict(sift_list, kmeans) 

    document_list = text_document(cluster_labels)
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(document_list) 

    knn = NearestNeighbors(n_neighbors=5, metric='cosine')# 5 because for queries I am finding 5 nearest images
    knn.fit(tfidf_matrix)

    indices = input(input_path, kmeans, vectorizer, knn, input_index)

    print("displaying")
    display_neighbors(indices, paths, input_path, input_index)

if __name__ == "__main__":
    main()
