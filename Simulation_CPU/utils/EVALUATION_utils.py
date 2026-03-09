import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

################################################### accuracy ##########################################################
def most_similar_group(scores):
    most_similar_groups = []
    for i in range(len(scores)):
        most_similar_groups.append(np.argmax(scores[i]))

    most_similar_groups = np.array(most_similar_groups)
    return most_similar_groups

def score_proba(scores):
    scores_proba = []
    for i in range(len(scores)):
        probas_temp = []
        total_score_temp = np.sum(scores[i])
        for j in range(len(scores[i])):
            probas_temp.append(scores[i][j]/total_score_temp)
        scores_proba.append(probas_temp)

    scores_proba = np.array(scores_proba)
    return scores_proba

def most_similar_group_proba(scores):
    most_similar_groups_proba = []
    for i in range(len(scores)):
        total_score_temp = np.sum(scores[i])
        max_temp = np.max(scores[i])
        most_similar_groups_proba.append(max_temp/total_score_temp)

    most_similar_groups_proba = np.array(most_similar_groups_proba)
    return most_similar_groups_proba

def calc_accuracy(test_labels, most_similar_group):
   
    
    correct = 0
    for i in range(len(test_labels)):
      if most_similar_group[i] == test_labels[i]:
        correct = correct + 1
    
    test = len(test_labels)
    
    accuracy = np.sum(np.array(most_similar_group) == test_labels[:test]) / test * 100
    print("accuracy: ", accuracy)

    
    return accuracy

def my_confusion_matrix(label_num, test_labels, most_similar_group):
    
    confusion = [[0] * label_num for _ in range(label_num)]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for image_num in range(len(test_labels)):
        confusion[test_labels[image_num]][most_similar_group[image_num]] += 1
        
    return confusion

def my_confusion_matrix_prob(label_num, test_labels, most_similar_group):
    confusion = [[0] * label_num for _ in range(label_num)]
    confusion_prob = [[0] * label_num for _ in range(label_num)]

    for image_num in range(len(test_labels)):
        confusion[test_labels[image_num]][most_similar_group[image_num]] += 1

    for i in range(len(confusion)):
        total_temp = sum(confusion[i])
        for j in range(len(confusion)):
            confusion_prob[i][j] = round(confusion[i][j]/total_temp, 4)

    return confusion_prob

def show_confusion_matrix(label_num, test_labels, most_similar_group, axis=0, show_count=False):
    if (show_count):
        confusion_matrix = my_confusion_matrix(label_num=label_num, test_labels=test_labels, most_similar_group=most_similar_group)
    else:
        confusion_matrix = my_confusion_matrix_prob(label_num=label_num, test_labels=test_labels, most_similar_group=most_similar_group)

    if (axis==0):
        confusion_matrix_transposed = [list(x) for x in zip(*confusion_matrix)]
        print(" "*12, end="")
        for i in range(len(confusion_matrix_transposed)):
            print(f"{class_mapping[i]:<12}", end="")
        print('(Actual)')
        
        for i in range(len(confusion_matrix_transposed)):
            print(f"{class_mapping[i]:<12}", end="")
            for j in range(len(confusion_matrix_transposed)):
                if (i==j):
                    score_temp = str(confusion_matrix_transposed[i][j])+" V"
                    print(f"{score_temp:<12}", end="")
                else:
                    print(f"{confusion_matrix_transposed[i][j]:<12}", end="")
            print()
        print('(Predict)')
    else:
        print(" "*12, end="")
        for i in range(len(confusion_matrix)):
            print(f"{class_mapping[i]:<12}", end="")
        print('(Predict)')
        
        for i in range(len(confusion_matrix)):
            print(f"{class_mapping[i]:<12}", end="")
            for j in range(len(confusion_matrix)):
                if (i==j):
                    score_temp = str(confusion_matrix[i][j])+" V"
                    print(f"{score_temp:<12}", end="")
                else:
                    print(f"{confusion_matrix[i][j]:<12}", end="")
            print()
        print('(Actual)')

    return None

def show_cf_matrix_heatmap(label_num, test_labels, most_similar_group, axis=0, show_count=False):
    if (show_count):
        confusion_matrix = my_confusion_matrix(label_num=label_num, test_labels=test_labels, most_similar_group=most_similar_group)
        title = 'Confusion Matrix: count'
        fmt = 'd'
    else:
        confusion_matrix = my_confusion_matrix_prob(label_num=label_num, test_labels=test_labels, most_similar_group=most_similar_group)
        title = 'Confusion Matrix: probability'
        fmt = '.2%'

    if (axis==0):
        confusion_matrix = [list(x) for x in zip(*confusion_matrix)]
        X_label = 'Actual'
        Y_label = 'Prediction'
    else:
        X_label = 'Prediction'
        Y_label = 'Actual'

    
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_matrix, annot=True, cbar=False, cmap='Blues', fmt=fmt, linewidth=0.3)
    plt.xlabel(X_label, fontsize=10)
    plt.ylabel(Y_label, fontsize=10)
    plt.show()


    return None

def clf_report(y_test, prediction, class_mapping):
    print(classification_report(y_test, prediction, target_names=class_mapping))
    return None

def calc_auc(y_test, probabilities):
    print("AUC: ", roc_auc_score(y_test, probabilities, average='weighted', multi_class='ovr'))
    return None

def plot_roc_curve(y_test, label_num, probabilities, class_mapping):

    # One-vs-Rest
    y_test_bin = label_binarize(y_test, classes=range(label_num))

    # ROC curve and AUC
    plt.figure(figsize=(8, 6))

    for i in range(label_num):
        # FPR, TPR
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        
        # ROC curve plotting
        plt.plot(fpr, tpr, label=f'Class {class_mapping[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    return None