import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import h5py


class FeatureMatcher:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher

    # type annotations
    def detect(self, img: np.ndarray, mask=None):
        kp, des = self.detector.detectAndCompute(img, mask)
        return kp, des

    def match(self, img1: np.ndarray, img2: np.ndarray):
        kp1, des1 = self.detect(img1)
        kp2, des2 = self.detect(img2)
        matches = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        return matches

    def match(self, des1: np.ndarray, des2: np.ndarray):
        matches = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        return matches

    def visualise(self, matches, img1: np.ndarray, kp1, img2: np.ndarray, kp2, title=''):
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append([m])
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow(title, img3)

if __name__ == "__main__":

    imgs = glob.glob('./img/*.jpg')

    img1 = cv.imread('bb_02.jpg', cv.IMREAD_GRAYSCALE)   # Reference Image
    img2 = cv.imread('./img/bb_04.jpg', cv.IMREAD_GRAYSCALE)

    # SIFT detector initialization
    sift = cv.xfeatures2d.SIFT_create()

    # AKAZE detector initialization
    akaze = cv.AKAZE_create()

    # BRISK detector initialization
    brisk = cv.BRISK_create()

    # ORB detector initialization
    orb = cv.ORB_create()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)

    # SIFT detector with FLANN matcher
    siftFlann = FeatureMatcher(sift, flann)

    # AKAZE detector with FLANN matcher
    akazeFlann = FeatureMatcher(akaze, flann)

    # BRISK detector with FLANN matcher
    briskFlann = FeatureMatcher(brisk, flann)

    # ORB detector with FLANN matcher
    orbFlann = FeatureMatcher(orb, flann)

    matchesSi = siftFlann.match(img1, img2)
    kpSi, desSi = siftFlann.detect(img1)
    kpSi1, desSi2 = siftFlann.detect(img2)
    siftFlann.visualise(matchesSi, img1, kpSi, img2, kpSi1, 'sift')

    matchesAk = akazeFlann.match(img1, img2)
    kpAk, desAk = akazeFlann.detect(img1)
    kpAk1, desAk1 = akazeFlann.detect(img2)
    akazeFlann.visualise(matchesAk, img1, kpAk, img2, kpAk1, 'akaze')

    cv.waitKey()



    """
    for image in imgs:
        img2 = cv.imread(image, cv.IMREAD_GRAYSCALE)
        matches = feature_matcher.match(img1, img2)
        kp1, des1 = feature_matcher.detect(img1)
        kp2, des2 = feature_matcher.detect(img2)
        feature_matcher.visualise(matches, img1, kp1, img2, kp2)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    
    plt.imshow(img3,),plt.show()
    """