import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob



class FeatureMatcher:
    """
    A class used to combine cv2 detector and cv2 matcher

    ...

    Attributes
    ----------
    detector
        a cv2 detector object
    matcher
        a cv2 DescriptorMatcher object

    Methods
    -------
    __init__(self, detector, matcher)
        A constructor method for FeatureMatcher class
    """
    def __init__(self, detector, matcher):
        """
        Parameters
        ----------
        detector
            a cv2 detector object
        matcher
            a cv2 DescriptorMatcher object
        """
        self.detector = detector
        self.matcher = matcher

    def detect(self, img: np.ndarray, mask=None):
        """Uses detector to detect keypoint and descriptors for image

            Uses class attribute detector to compute and return keypoints and descriptors for image given by numpy ndarray img

            Parameters
            ----------
            img : np.ndarray
                Numpy array of given image
            mask : optional
                Mask specifying where to look for keypoints, default is None
        """

        kp, des = self.detector.detectAndCompute(img, mask)
        return kp, des

    def match(self, img1: np.ndarray, img2: np.ndarray):
        """This method uses the detect method to get keypoints and descriptors for given images, it then matches them using matcher knnMatch method

            Method calculates descriptors for images img1 and img2 given by numpy ndarrays. It then takes those descriptors and uses knnMatch method of matcher attribute.
            This method returns k best matches of descriptors for given images. k is always set to 2.

            Parameters
            ----------
            img1 : np.ndarray
                Numpy array of the first image
            img2 : np.ndarray
                Numpy array of the second image
        """

        kp1, des1 = self.detect(img1)
        kp2, des2 = self.detect(img2)
        matches = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        return matches

    def match(self, des1: np.ndarray, des2: np.ndarray):
        """This is an overloaded method, it differs from the above method by parameters it accepts

            Method takes two sets of descriptors des1 and des2 and uses matcher knnMatch method to compare them and find k best matches. k is set to 2.
            Parameters
            ----------
            des1 : np.ndarray
                the first set of descriptors
            des2 : np.ndarray
                the second set of descriptors
        """
        matches = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        return matches

    def visualise(self, matches, img1: np.ndarray, kp1, img2: np.ndarray, kp2, title=''):
        """This method visualises matched keypoints between two given images

            Parameters
            ----------
            matches
                Takes matched descriptors of two images given by match method
            img1 : np.ndarray
                Numpy array of the first image
            img2 : np.ndarray
                Numpy array of the second image
            kp1
                Set of keypoints for the first image
            kp2
                Set of keypoints for the second image
            title : , optional
                Title of image, default title is an empty string ''
        """

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