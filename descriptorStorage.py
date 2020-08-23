import h5py
import json
from feature_match import FeatureMatcher
import cv2 as cv
import glob


def init_db(filename: str, path_to_images: str, matcher : FeatureMatcher, mapping_file : str = 'data.txt', des_dataset: str = 'DatasetDescriptors', kp_dataset: str = 'DatasetKeypoints'):
    """Initializes h5py file or overwrites the existing one, resets image mapping

        Creates new h5py file with given name via filename parameter. This file will always have two datasets, descriptor dataset
        named via des_dataset parameter and keypoint dataset named via kp_dataset parameter. Mapping file must be given via mapping_file parameter.
        Method then detects and computes descriptors and keypoints of all images in path_to_images directory. This is done using cv2 detector passed by matcher parameter

        Parameters
        ----------
        filename : str
            Name of the new h5py file
        path_to_images : str
            Path to directory with images
        matcher : FeatureMatcher
            FeatureMatcher class object with cv2 detector and cv2 matcher
        mapping_file : str , optional
            Name of the mapping file containing json with images and positions of their keypoints and descriptors in datasets
        des_dataset :str , optional
            Name of the dataset containing descriptors, default is DatasetDescriptors
        kp_dataset : str, optional
            Name of the dataset containing keypoints, default is DatasetKeypoints
    """

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
    """Updates existing h5py file with new image descriptors and keypoints

        Updates existing h5py file to contain new images added to path_to_images directory. This file will always have two datasets, descriptor dataset
        named via des_dataset parameter and keypoint dataset named via kp_dataset parameter. Datasets wil lbe created if not already contained in h5py file.
        Mapping file must be given via mapping_file parameter. Method then detects and computes descriptors and keypoints of new images in path_to_images directory.
        This is done using cv2 detector passed by matcher parameter

        Parameters
        ----------
        filename : str
            Name of the existing h5py file
        path_to_images : str
            Path to directory with images
        matcher : FeatureMatcher
            FeatureMatcher class object with cv2 detector and cv2 matcher
        mapping_file : str , optional
            Name of the mapping file containing json with images and positions of their keypoints and descriptors in datasets
        des_dataset :str , optional
            Name of the dataset containing descriptors, default is DatasetDescriptors
        kp_dataset : str, optional
            Name of the dataset containing keypoints, default is DatasetKeypoints
    """

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
    """Looks into existing h5py file and returns descriptors for given image

        This method looks into h5py file given by filename parameter and looks for descriptors in dataset des_dataset.
        It gets boundaries of image descriptors from mapping_file json for image given by parameter image_path.

        Parameters
        ----------
        filename : str
            Name of existing h5py file
        image_path : str
            Specifies the path to the image
        mapping_file : str , optional
            Specifies the name of the mapping file containing boundaries of descriptors for given image
        des_dataset : str , optional
            Specifies the dataset that will be searched for the descriptors
    """

    with h5py.File(filename, 'r') as f:
        with open(mapping_file) as jfile:
            mapping = json.load(jfile)
        des_set = f[des_dataset]
        descriptor = des_set[mapping[image_path][0]:mapping[image_path][1], :]
        return descriptor


def get_kp(filename: 'str', image_path: 'str', mapping_file : str = 'data.txt', kp_dataset: str = 'DatasetKeypoints'):
    """Looks into existing h5py file and returns keypoints for given image

        This method looks into h5py file given by filename parameter and looks for keypoints in dataset kp_dataset.
        It gets boundaries of image keypoints from mapping_file json for image given by parameter image_path.

        Parameters
        ----------
        filename : str
            Name of existing h5py file
        image_path : str
            Specifies the path to the image
        mapping_file : str , optional
            Specifies the name of the mapping file containing boundaries of keypoints for given image
        des_dataset : str , optional
            Specifies the dataset that will be searched for the keypoints
    """

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
    """This method empties the mapping file"""

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



