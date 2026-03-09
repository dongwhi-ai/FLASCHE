import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import os
import struct
from PIL import Image
from tqdm import tqdm


def data_selection_digits(train_size, test_size, random_seed):
    digits = load_digits()
    X_data = digits.data.reshape(-1, 8, 8)
    y_data = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=train_size, test_size=test_size, random_state=random_seed, stratify=y_data)

    # test_images, test_labels count check
    for i in range(np.max(y_test) + 1):
        print(f"Test images: Class {i} - {len(X_test[y_test == i])} images")
    print()
    
    # train_images, train_labels count check
    for i in range(np.max(y_train) + 1):
        print(f"Train images: Class {i} - {len(X_train[y_train == i])} images")
    print()
    
    # total image count check
    for i in range(np.max(y_test) + 1):
        test_images_count = len(X_test[y_test == i])
        train_images_count = len(X_train[y_train == i])
        total_images_count = test_images_count + train_images_count
        print(f"Total images: Class {i} - {total_images_count} images")
    print()

    return X_train, X_test, y_train, y_test

def load_mnist(data_dir, kind='train'):
    labels_path = os.path.join(data_dir,'MNIST/%s-labels.idx1-ubyte' %kind)
    images_path = os.path.join(data_dir,'MNIST/%s-images.idx3-ubyte' %kind)

    with open(labels_path,'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path,'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),784)

    return images, labels

def data_selection_mnist(data_dir, train_size, test_size, random_seed):
    print("train data downloading...")
    X_train, y_train = load_mnist(data_dir=data_dir, kind='train')
    print("test data downloading...")
    X_test, y_test = load_mnist(data_dir=data_dir, kind='t10k')


    """
    fig,ax = plt.subplots( nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.ravel()
    for i in range(10):
        img = X_train[y_train == i ][0].reshape(28,28)
        ax[i].imshow(img,cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    """

    X_train = X_train.reshape(60000, 28, 28)
    X_test = X_test.reshape(10000, 28, 28)
    X_data = np.concatenate((X_train, X_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=train_size, test_size=test_size, random_state=random_seed, stratify=y_data)
    
    # test_images, test_labels count check
    for i in range(np.max(y_test) + 1):
        print(f"Test images: Class {i} - {len(X_test[y_test == i])} images")
    print()
    
    # train_images, train_labels count check
    for i in range(np.max(y_train) + 1):
        print(f"Train images: Class {i} - {len(X_train[y_train == i])} images")
    print()
    
    # total image count check
    for i in range(np.max(y_test) + 1):
        test_images_count = len(X_test[y_test == i])
        train_images_count = len(X_train[y_train == i])
        total_images_count = test_images_count + train_images_count
        print(f"Total images: Class {i} - {total_images_count} images")
    print()

    return X_train, X_test, y_train, y_test

def load_all_original_MNIST(data_dir):
    X_train, y_train = load_mnist(data_dir=data_dir, kind='train')
    X_test, y_test = load_mnist(data_dir=data_dir, kind='t10k')
    X_train = X_train.reshape(60000, 28, 28)
    X_test = X_test.reshape(10000, 28, 28)

    # test_images, test_labels count check
    for i in range(np.max(y_test) + 1):
        print(f"Test images: Class {i} - {len(X_test[y_test == i])} images")
    print()
    
    # train_images, train_labels count check
    for i in range(np.max(y_train) + 1):
        print(f"Train images: Class {i} - {len(X_train[y_train == i])} images")
    print()
    
    # total image count check
    for i in range(np.max(y_test) + 1):
        test_images_count = len(X_test[y_test == i])
        train_images_count = len(X_train[y_train == i])
        total_images_count = test_images_count + train_images_count
        print(f"Total images: Class {i} - {total_images_count} images")
    print()

    return X_train, X_test, y_train, y_test

def load_image_inout(data_dir, position, kind='train'):
    images = []
    labels = []
    data_dir = os.path.join(data_dir, position)
    for class_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_label)
        if os.path.isdir(class_dir):
            if (class_label in class_mapping):
                label = class_mapping.index(class_label)
            else:
                label = None
            if label is None:
                continue
                
            
            images_path = os.path.join(class_dir, kind)
            # load image
            for filename in tqdm(os.listdir(images_path)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(images_path, filename)
                    image = Image.open(image_path)
                    image = np.array(image)
                    images.append(image)
                    labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def data_selection_inout(data_dir, data_use):    
    print("train data downloading...")
    X_train, y_train = load_image_inout(data_dir=data_dir, position=data_use, kind='train')
    print("test data downloading...")
    X_test, y_test = load_image_inout(data_dir=data_dir, position=data_use, kind='test')

    # test_images, test_labels count check
    for i in range(np.max(y_test) + 1):
        print(f"Test images: Class {i} - {len(X_test[y_test == i])} images")
    print()
    
    # train_images, train_labels count check
    for i in range(np.max(y_train) + 1):
        print(f"Train images: Class {i} - {len(X_train[y_train == i])} images")
    print()
    
    # total image count check
    for i in range(np.max(y_test) + 1):
        test_images_count = len(X_test[y_test == i])
        train_images_count = len(X_train[y_train == i])
        total_images_count = test_images_count + train_images_count
        print(f"Total images: Class {i} - {total_images_count} images")
    print()

    return X_train, X_test, y_train, y_test