import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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