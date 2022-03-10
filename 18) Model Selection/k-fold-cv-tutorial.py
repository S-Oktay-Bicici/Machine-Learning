from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# import ettiğimiz datayı x y değerlerine atıyoruz
iris = load_iris()

x = iris.data
y = iris.target

# normalization
x = (x-np.min(x))/(np.max(x)-np.min(x))

# train test split olarak ayırıyoruz 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3)

# knn model import edip, k değerini 3 olarak belirliyoruz
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)  # k = n_neighbors

# K fold CrossValidation değerini belirliyoruz K = 10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn, X = x_train, y= y_train, cv = 10)
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))

# test için ayırdığımız kısım ile test ederek doğruluğuna bakıyoruz 
knn.fit(x_train,y_train)
print("test accuracy: ",knn.score(x_test,y_test))


# grid search cross validation for knn

from sklearn.model_selection import GridSearchCV

#knn algoritması için k değerini 50 farklı şekilde deniyourz
grid = {"n_neighbors":np.arange(1,50)}
knn= KNeighborsClassifier()

#değişen k değerlerinin sonuçları için grid yapısını kullanıyoruz 
knn_cv = GridSearchCV(knn, grid, cv = 10)  # GridSearchCV
knn_cv.fit(x,y)

# print hyperparameter KNN algoritmasindaki K degeri

# 1-50 arası seçmem gereken k değeri 
print("tuned hyperparameter K: ",knn_cv.best_params_)
# seçmem gereken k değeri elde ettiğim doğruluk oranı
print("tuned parametreye gore en iyi accuracy (best score): ",knn_cv.best_score_)

""" Grid search CV with logistic regression

x = x[:100,:]
y = y[:100] 

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)

"""







































