import pickle
import numpy as np
import rasterio

def directories():
    PATH_INPUTS = '/tesis/inputs/'
    PATH_OUTPUTS = '/tesis/outputs/'
    return PATH_INPUTS, PATH_OUTPUTS


def read_image(file_name):
    with rasterio.open(file_name) as src:
        array = src.read()
    return np.array(array,dtype=np.float32)


def read_clustering(file_name):
    with rasterio.open(file_name) as src:
        array = src.read()[0]
    return np.array(array,dtype=np.int16)


def save_in_file(X, with_name):
    with open('data.pickle', 'wb') as f:
        pickle.dump(X, open('%s.pkl' % with_name, 'wb'))


def load_from_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def find_and_replace(new_image, old_label, new_label):
    new_image[new_image == old_label] = new_label


def save_image(source_file, destination_file, data):
    _save_gtiff(source_file, destination_file, data, "image")


def save_clustering(source_file, destination_file, data):
    _save_gtiff(source_file, destination_file, data, "clustering")


def _save_gtiff(source_file, destination_file, data, type):
    if type == "image":
        dtype = np.float32
    elif type == "clustering":
        dtype = np.int16

    with rasterio.open(source_file) as src:
        crs = src.crs
        transform = src.transform
        aux_image = np.array(src.read())[0].copy()
        height = aux_image.shape[0]
        width = aux_image.shape[1]

    with rasterio.open(destination_file,'w',driver='GTiff',height=height,width=width,count=1,dtype=dtype,crs=crs,transform=transform) as dst:
        dst.write(data.astype(dtype), 1)
