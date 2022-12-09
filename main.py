from os.path import exists
import KMeans
import numpy as np
import cv2 as cv
import log
import pickle
import sys
import threading

def do_fill(aux,x,y, current_cluster = [], first_call = False):
    """
    Follows pixels in an edge, transforms to 0 visited pixels, adds visited pixels to an array, and continues filling the 8-connected neighbors recursively
    """
    
    if x < 0 or y<0 or x>=aux.shape[0] or y >= aux.shape[1]:
        return
    # append the coordinates to the cluster
    current_cluster.append((y,x))
    # this pixel will not be visited again in next iterations
    aux[x][y] = np.array([0,0,0])

    # perform the fill in the 8-connected neighbors of the current pixel
    for i in range(-1,2):
        for j in range(-1,2):
            if np.array_equal(aux[x+i][y+j], np.array([255, 255, 255])):
                do_fill(aux, x+i, y+j, current_cluster)
    if not first_call:
        return
    return current_cluster


def cluster_edges(edges_img:np.ndarray,clusters_ret):
    """
    Looks for white pixels and clusters those that belong to the same edge
        Parameters:
            edges_img : image with detected edges
        Returns:
            clusters : array of clusters containing the coordinates of pixels that belong to that cluster
    """
    w, h, c = edges_img.shape
    aux = edges_img.copy()
    white = np.array([255,255,255])
    clusters = []
    for x in range(w):
        for y in range(h):
            if np.array_equal(aux[x][y],white):
                clusters.append(do_fill(aux, x, y, [],True))
    clusters_ret[0] = clusters.copy()
    return clusters

def segmentate(img:np.ndarray, segments: int, centroids: np.ndarray = None) -> np.ndarray:
    """
    Segmentates an image by colors using k-means
        Parameters:
            img: image to perform the segmentation
            segments: number of colors to segment the image
        Returns:
            img_processed: image segmentated in n-segments clusters
    """
    np.random.seed()
    w, h, c = img.shape
    # xv,yv = coords_matrix(w, h)
    img_preprocessed = np.reshape(img,(w * h, c))
    img_processed = np.zeros((w, h, c), dtype=np.uint8)

    kmeans = KMeans.KMeans(segments)
    kmeans.fit(img_preprocessed, centroids)
    
    for x in range(w):
        for y in range(h):
            sample = img[x][y]
            cluster_predicted = kmeans.predict(sample)
            img_processed[x][y] = np.floor(kmeans.centroids[cluster_predicted])
    return img_processed


def resize_img(src: np.ndarray, scale_percent: int) -> np.ndarray:
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    dsize = (width, height)

    output = cv.resize(src, dsize)
    return output

def filter_maximize_red(img: np.ndarray, color: np.ndarray):
    w,h,c = img.shape
    img_processed = np.zeros((w, h, c), dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            b,g,r = img[x][y]
            if r >= color[0] and g <= color[1] and b <= color[2]:
                img_processed[x][y] = 255
    return img_processed


if __name__ == "__main__":
    sys.setrecursionlimit(10**9)
    threading.stack_size(10**8)
    img = cv.imread("Jit1.jpg")
    """
    if not exists("./resized.png"):
        resized = resize_img(img, 100)
        cv.imwrite('./resized.png', resized)
    resized = cv.imread("./resized.png")
    cv.imshow("Resized",resized)
    """
    if not exists("./blurred.png"):
        blurred = cv.GaussianBlur(img,(25,25),20)
        cv.imwrite("./blurred.png",blurred)
    blurred = cv.imread("./blurred.png")
    cv.imshow("Blurred",blurred)

    if not exists("./segmentated.png"):
        segmentated = segmentate(blurred, 5)
        cv.imwrite("./segmentated.png",segmentated)
    segmentated = cv.imread("./segmentated.png")
    cv.imshow("Segmentated",segmentated)
    
    if not exists("./filtered_color.png"):
        filtered_color = filter_maximize_red(segmentated, np.array([100, 50, 50]))
        cv.imwrite("./filtered_color.png", filtered_color)
    filtered_color = cv.imread("./filtered_color.png")
    cv.imshow("Filtered", filtered_color)
    
    if not exists("./log.png"):
        edges_img = log.algoritmoLaplacianoGauss(filtered_color, 5, 1, 1)
        cv.imwrite("./log.png",edges_img)
    edges_img = cv.imread("./log.png")
    cv.imshow("LoG",edges_img)

    
    km = KMeans.KMeans(1)
    if not exists("./clustered_edges.pkl"):
        clusters = [None] * 2
        t = threading.Thread(target=cluster_edges, args=(edges_img,clusters))
        t.start()
        t.join()
        clusters = clusters[0]
        sorted_clusters = []
        print(len(clusters))
        for cluster in clusters:
            if (len(cluster) > 2000):
                sorted_clusters.append(sorted(cluster, key=lambda x: x[0]))
        file = open("./clustered_edges.pkl","wb")
        pickle.dump(sorted_clusters,file)
        file.close()
    file = open("./clustered_edges.pkl","rb")
    sorted_clusters = pickle.load(file)
    file.close()

    if not exists("./distances.png"):
        distances_img = img.copy()
        for i,cluster in enumerate(sorted_clusters):
            distances_img = cv.line(
                distances_img, cluster[0], cluster[-1], (0, 255, 0), 2)
            
            middle = (np.array(cluster[-1]) - np.array(cluster[0])) / 2
            middle = np.array(cluster[0]) + middle
            distances_img = cv.putText(
                distances_img, f"{i+1}", (int(middle[0]),int(middle[1])), cv.FONT_HERSHEY_SIMPLEX,5,(255,0,0),5,cv.LINE_AA)
        cv.imwrite("./distances.png",distances_img)
    distances_img = cv.imread("./distances.png")
    for i,cluster in enumerate(sorted_clusters):
        print(f"Segmento {i+1} -> distancia desde {cluster[0]} hasta {cluster[-1]} : {km.euclidean_distance(np.array(cluster[0]),np.array(cluster[-1])):.0f} pixeles")
    cv.imshow("Distancias", distances_img)

    cv.waitKey()

