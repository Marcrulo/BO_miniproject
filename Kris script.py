import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

def importdataset():
    training_set = datasets.load_wine(root='./data', train=True, download=True, transform=None)
    test_set = datasets.load_wine(root='./data', train=False, download=True, transform=None)

    Xtrain = training_set.data.numpy().reshape(-1, 28 * 28)
    Xtest = test_set.data.numpy().reshape(-1, 28 * 28)

    ytrain = training_set.targets.numpy()
    ytest = test_set.targets.numpy()

    return X, y

def KnnOnData(X, y):
    #Create KNN Object.
    knn = KNeighborsClassifier()

    #Split data into training and testing.
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    #Training the model.
    knn.fit(x_train, y_train)

    #Predict test data set.
    y_pred = knn.predict(x_test)

    #Checking performance our model with classification report.
    print(classification_report(y_test, y_pred))





if __name__ == "__main__":
    X, y = importdataset()
    KnnOnData(X, y)

