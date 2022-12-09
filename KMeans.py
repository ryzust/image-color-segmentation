import numpy as np
import cv2 as cv
import math

class KMeans:
    def __init__(self, nClusters):
        self.nClusters = nClusters
        self.clusters = []
        np.random.seed()

    def euclidean_distance(self,p1, p2):
        n = p1.shape[0]
        sum_squares = 0
        for i in range(n):
            sum_squares += (p1[i] - p2[i]) ** 2
        return math.sqrt(sum_squares)

    def cluster_samples(self,dataset:np.ndarray):
        """Assigns all the samples in the dataset to a cluster using euclidean distance"""
        self.clusters = []
        for nCluster in range(self.nClusters):
            self.clusters.insert(nCluster, [])
        for sample in dataset:
            most_closer_nCluster = self.predict(sample)
            self.clusters[most_closer_nCluster].append(sample)
    
    def recalculate_centroids(self):
        """Recalculates centroids based on the average of each feature for each cluster"""
        for nCluster in range(self.nClusters):
            cluster = np.array(self.clusters[nCluster])
            tmp_centroid = []
            if len(cluster) <= 0:
                continue
            for nFeature in range(self.nFeatures):
                feature_array = cluster[:, nFeature]
                tmp_centroid.append(np.average(feature_array))
            self.centroids[nCluster] = np.floor(tmp_centroid)
    
    def assign_random_centroids(self,dataset:np.ndarray):
        """Assigns new centroids choosing n-cluster random samples and taking it as the new centroids"""
        nSamples, _ = dataset.shape
        self.centroids = np.zeros((self.nClusters, self.nFeatures))
        for nCluster in range(self.nClusters):
            rnd = np.random.randint(0, nSamples)
            self.centroids[nCluster] = dataset[rnd]

    def fit(self,dataset:np.ndarray, initial_centroids: np.ndarray = None):
        """Trains a K-Means model using the samples given in the dataset"""
        nSamples, nFeatures = dataset.shape
        self.nFeatures = nFeatures

        if initial_centroids is None:
            self.assign_random_centroids(dataset)
        else:
            nCentroids, nFeaturesCentroids = initial_centroids.shape
            if nCentroids != self.nClusters or nFeatures != nFeaturesCentroids:
                self.assign_random_centroids(dataset)
            else:
                self.centroids = initial_centroids


        i = 0
        tmpCentroids = np.array([])
        while (not np.array_equal(tmpCentroids, self.centroids)) and i < 50:           
            print(i)
            tmpCentroids = self.centroids.copy()
            self.cluster_samples(dataset)
            self.recalculate_centroids()
            i += 1

            

    def predict(self,sample):
        """Assign the sample to a cluster and returns the number of cluster which is most closer to the sample"""
        most_closer_distance = float('inf')
        most_closer_nCluster = -1
        for nCluster in range(self.nClusters):
            centroid = self.centroids[nCluster]
            tmp_distance = self.euclidean_distance(centroid, sample)
            if tmp_distance < most_closer_distance:
                most_closer_distance = tmp_distance
                most_closer_nCluster = nCluster
    
        return most_closer_nCluster
        
                

if __name__ == "__main__":
    img = cv.imread("Jit1.jpg")
    img = cv.resize(img,())
    w,h,c = img.shape
    img_preprocessed = np.reshape(img,(w*h,c))
    img_processed = np.zeros((w,h,c), dtype=np.uint8)
    kmeans = KMeans(2)
    kmeans.fit(img_preprocessed)
    for x in range(w):
        for y in range(h):
            sample = img[x][y]
            cluster_predicted = kmeans.predict(sample)
            img_processed[x][y] = np.floor(kmeans.centroids[cluster_predicted])
    cv.imshow("Kmeans",img_processed)
    cv.waitKey()
    cv.imwrite("./segmentated.png")

