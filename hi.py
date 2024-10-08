import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def knn_predict(Xtrain, Ytrain, Xtest, k):
    predictions = []
    for i in range(len(Xtest)):
        class_count = [0, 0, 0]
        distances = []
        for j in range(len(Xtrain)):
            distances.append(np.sqrt(np.sum(np.square(Xtest[i] - Xtrain[j]))))
        distances = np.array(distances)
        sorted_indices = np.argsort(distances)
        nearest_labels = Ytrain[sorted_indices[:k]]
        for i in range(k):
            if nearest_labels[i] == 0:
                class_count[0] += 1
            elif nearest_labels[i] == 1:
                class_count[1] += 1
            elif nearest_labels[i] == 2:
                class_count[2] += 1
        predictions.append(np.argmax(class_count))
    return predictions

def kfold_cross_validation(X, y, k, folds):
    fold_size = len(X) // folds
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    accuracies = []
    
    for fold in range(folds):
        start, end = fold * fold_size, (fold + 1) * fold_size
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))
        
        Xtrain, Xtest = X[train_indices], X[test_indices]
        ytrain, ytest = y[train_indices], y[test_indices]
        
        ypred = knn_predict(Xtrain, ytrain, Xtest, k)
        
        accuracy = np.mean(ypred == ytest)
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)

def main():
    
    iris = pd.read_csv('Lab_A7_iris.data.csv')
    X = iris.iloc[:, :-1].values
    Y = iris.iloc[:, -1].values
    
    for i in range(len(Y)):
        if Y[i] == 'Iris-setosa':
            Y[i] = 0
        elif Y[i] == 'Iris-versicolor':
            Y[i] = 1
        elif Y[i] == 'Iris-virginica':
            Y[i] = 2
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=246)
    
    # Perform 4-fold cross-validation to find the best K value for KNN
    accuracies = []
    std_devs = []
    for k in range(1, 10):
        mean_acc, std_acc = kfold_cross_validation(Xtrain, Ytrain, k, 4)
        accuracies.append(mean_acc)
        std_devs.append(std_acc)

    best_k = np.argmax(accuracies) + 1
    
    # Report the test accuracy using the best K value
    Ypred_test = knn_predict(Xtrain, Ytrain, Xtest, best_k)
    test_accuracy = np.mean(Ypred_test == Ytest)

    print("---- Results ----\n")
    print(f"Best K value (from cross-validation): {best_k}")
    print(f"Validation Accuracy (Best K): {accuracies[best_k-1]:.4f}")
    print(f"Validation Standard Deviation (Best K): {std_devs[best_k-1]:.4f}")
    print(f"Test Accuracy (Best K): {test_accuracy:.4f}\n")

    # Summary of all accuracies and standard deviations for each K
    print("---- Summary of Validation Accuracies ----")
    for i in range(1, 10):
        print(f"K = {i}, Accuracy = {accuracies[i-1]:.4f}, Standard Deviation = {std_devs[i-1]:.4f}")

if __name__ == "__main__":
    main()
