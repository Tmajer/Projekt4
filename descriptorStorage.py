import h5py
import json
from feature_match import FeatureMatcher
import cv2 as cv
import glob


def init_db(filename: str, path_to_images: str, matcher : FeatureMatcher, mapping_file : str = 'data.txt', des_dataset: str = 'DatasetDescriptors', kp_dataset: str = 'DatasetKeypoints'):

    reset_db(mapping_file)

    with h5py.File(filename, 'w') as f:
        if not des_dataset in f.keys():
            des_set = f.create_dataset(des_dataset, (500000, 128), maxshape=(None, 128))
        des_set = f[des_dataset]
        if not kp_dataset in f.keys():
            kp_set = f.create_dataset(kp_dataset, (500000, 7), maxshape=(None, 7))
        kp_set = f[kp_dataset]

        with open(mapping_file) as jfile:
            mapping = json.load(jfile)

        imgs = glob.glob(path_to_images)
        if not 'start_index_for_new' in mapping.keys():
            mapping['start_index_for_new'] = 0
        start_index = mapping['start_index_for_new']

        for image in imgs:
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
            if (not image in mapping) or not mapping[image]:
                kp, des = matcher.detect(img)
                des_count, _ = des.shape
                keypoints = []
                for point in kp:
                    temp = (point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave, point.class_id)
                    keypoints.append(temp)
                end_index = start_index + des_count
                des_set[start_index:end_index, ...] = des
                kp_set[start_index:end_index, ...] = keypoints

                mapping[image] = (start_index, end_index)
                start_index += des_count
                mapping['start_index_for_new'] = start_index

        with open('data.txt', 'w') as out:
            json.dump(mapping, out)


def update_db(filename: str, path_to_images: str, matcher : FeatureMatcher, mapping_file : str = 'data.txt', des_dataset: str = 'DatasetDescriptors', kp_dataset: str = 'DatasetKeypoints'):
    with h5py.File(filename, 'a') as f:
        if not des_dataset in f.keys():
            des_set = f.create_dataset(des_dataset, (500000, 128), maxshape=(None, 128))
        des_set = f[des_dataset]
        if not kp_dataset in f.keys():
            kp_set = f.create_dataset(kp_dataset, (500000, 7), maxshape=(None, 7))
        kp_set = f[kp_dataset]

        with open(mapping_file) as jfile:
            mapping = json.load(jfile)

        imgs = glob.glob(path_to_images)
        if not 'start_index_for_new' in mapping.keys():
            mapping['start_index_for_new'] = 0
        start_index = mapping['start_index_for_new']

        for image in imgs:
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
            if (not image in mapping) or not mapping[image]:
                kp, des = matcher.detect(img)
                des_count, _ = des.shape
                keypoints = []
                for point in kp:
                    temp = (point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave, point.class_id)
                    keypoints.append(temp)
                end_index = start_index + des_count
                des_set[start_index:end_index, ...] = des
                kp_set[start_index:end_index, ...] = keypoints

                mapping[image] = (start_index, end_index)
                start_index += des_count
                mapping['start_index_for_new'] = start_index

        with open('data.txt', 'w') as out:
            json.dump(mapping, out)


def get_des(filename: 'str', image_path: 'str', mapping_file : str = 'data.txt', des_dataset: str = 'DatasetDescriptors'):
    with h5py.File(filename, 'r') as f:
        with open(mapping_file) as jfile:
            mapping = json.load(jfile)
        des_set = f[des_dataset]
        descriptor = des_set[mapping[image_path][0]:mapping[image_path][1], :]
        return descriptor


def get_kp(filename: 'str', image_path: 'str', mapping_file : str = 'data.txt', kp_dataset: str = 'DatasetKeypoints'):
    with h5py.File(filename, 'r') as f:
        with open(mapping_file) as jfile:
            mapping = json.load(jfile)
        kpset = f[kp_dataset]
        kp = []

        for i in range(mapping[image_path][0], mapping[image_path][1]):
            temp = cv.KeyPoint(x = kpset[i][0], y=kpset[i][1], _size=kpset[i][2], _angle=kpset[i][3],  _response=kpset[i][4], _octave=kpset[i][5], _class_id=kpset[i][6])
            kp.append(temp)

        return kp


def reset_db(mapping_file : str):
    with open(mapping_file, 'w') as out:
            mapping = {}
            json.dump(mapping, out)


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
    # init_db('siftFile2.hdf5', './img/*.jpg', siftFlann) # FOR FIRST RUN ONLY
    update_db('siftFile2.hdf5', './img/*.jpg', siftFlann) # EVERY OTHER RUN

    with h5py.File('siftFile.hdf5', 'w') as f:
        if not 'siftDatasetDescriptors' in f.keys():
            des_set = f.create_dataset('siftDatasetDescriptors', (500000, 128), maxshape=(None, 128))
        des_set = f['siftDatasetDescriptors']
        if not 'siftDatasetKeypoints' in f.keys():
            key_set = f.create_dataset('siftDatasetKeypoints', (500000, 7), maxshape=(None, 7))
        key_set = f['siftDatasetKeypoints']
        start_index_des = 0

        with open('test.txt') as jfile:
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
                mapping['start_index_for_new'] = start_index_des
    with open('test.txt', 'w') as out:
            json.dump(mapping, out)

    for image in imgs:
        des_method = get_des('siftFile2.hdf5', image)
        des_nomethod = get_des('siftFile.hdf5', image, mapping_file='test.txt', des_dataset= 'siftDatasetDescriptors')
        kp_method = get_kp('siftFile2.hdf5', image)
        kp_nomethod = get_kp('siftFile.hdf5', image, mapping_file='test.txt', kp_dataset= 'siftDatasetKeypoints')

        print('Some descriptors')
        print(des_method[1][0:5])
        print(des_nomethod[1][0:5])

        print('Some keypoints')
        print(kp_method[0].pt, kp_method[0].size, kp_method[0].angle)
        print(kp_nomethod[0].pt,kp_nomethod[0].size, kp_nomethod[0].angle)

    reset_db('test.txt')



