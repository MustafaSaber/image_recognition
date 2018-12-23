from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm

from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 6)
    print(dog_files.shape)
    print(dog_targets.shape)
    return dog_files, dog_targets


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(32, 32))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# load train, test, and validatioyn datasets
# train_files, train_targets = load_dataset('H:\\courses\\Fourth\\FT\\Machine Learning\\Project\\Dataset-20181222T194147Z-001\\Dataset')
# tensors = paths_to_tensor(train_files)
# print(tensors.shape)