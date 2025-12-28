import cv2
import numpy as np
from sklearn.cluster import KMeans


class ColorSegmentation:
    def __init__(self, image_path, n_colors=5):
        self.image_path = image_path
        self.n_colors = n_colors
        self.kmeans = None
        self.centroids = None
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

    def segment(self):
        h, w, c = self.image.shape
        data = self.image.reshape((-1, 3))
        self.kmeans = KMeans(n_clusters=self.n_colors, random_state=0)
        labels = self.kmeans.fit_predict(data)
        self.centroids = self.kmeans.cluster_centers_.astype(np.uint8)
        segmented = self.centroids[labels].reshape((h, w, 3))
        return segmented


if __name__ == "__main__":
    _img_path_in = "../assets/images/background.jpg"
    _img_path_out = "../assets/images/segmented_output.jpg"
    _n_colors = 5
    _segmenter = ColorSegmentation(image_path=_img_path_in, n_colors=_n_colors)
    segmented_img = _segmenter.segment()
    cv2.imwrite(_img_path_out, segmented_img)
    print(f"Segmented image saved as {_img_path_out}")
