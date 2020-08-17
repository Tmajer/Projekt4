import h5py
import json
from feature_match import FeatureMatcher
import cv2 as cv
import glob

if __name__ == '__main__':
    image1 = './img\\bb_02.jpg'
    img1 = cv.imread('bb_02.jpg', cv.IMREAD_GRAYSCALE)  # Reference Image
    imgs = glob.glob('./img/*.jpg')

    sift = cv.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    siftFlann = FeatureMatcher(sift, flann)
    with h5py.File('siftFile.hdf5', 'w') as f:
        if not 'siftDatasetDescriptors' in f.keys():
            des_set = f.create_dataset('siftDatasetDescriptors', (500000, 128), maxshape=(None, 128))
        des_set = f['siftDatasetDescriptors']
        if not 'siftDatasetKeypoints' in f.keys():
            key_set = f.create_dataset('siftDatasetKeypoints', (500000, 7), maxshape=(None, 7))
        key_set = f['siftDatasetKeypoints']
        start_index_des = 0

        with open('data.txt') as jfile:
            mapping = json.load(jfile)

        for image in imgs:
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
            if (not image in mapping) or not mapping[image]:
                kp, des = siftFlann.detect(img)
                des_count, _ = des.shape
                keypoints = []
                for point in kp:
                    temp = (point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave, point.class_id)
                    keypoints.append(temp)
                end_index_des = start_index_des + des_count
                des_set[start_index_des:end_index_des, ...] = des
                key_set[start_index_des:end_index_des, ...] = keypoints

                mapping[image] = (start_index_des, end_index_des)
                start_index_des += des_count

with open('data.txt', 'w') as out:
            json.dump(mapping, out)