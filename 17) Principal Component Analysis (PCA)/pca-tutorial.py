#datamızı import ediyrouz ve kütüphaneleri ekliyoruz 
from sklearn.datasets import load_iris
import pandas as pd

# kütüphaneden aldıpımız datayı dataframe haline getiriyoruz
iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns = feature_names)
df["sinif"] = y

x = data

# PCA kütüphanesini eklyoruz
from sklearn.decomposition import PCA
# datayı kaç boyuta indirgeyeceğimiz belirliyoruz 
pca = PCA(n_components = 2, whiten= True )  # whitten = normalize
pca.fit(x)

# x datasını 2 boyuta döüştürüyorum
x_pca = pca.transform(x)

#gerçek datayı ne kadar temsil ettiğine bakıyoruz
print("variance ratio: ", pca.explained_variance_ratio_)

#datamın ne kadaraını kaybettiğimizi öğreniyoruz 
print("sum: ",sum(pca.explained_variance_ratio_))

#%% 2D

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
























