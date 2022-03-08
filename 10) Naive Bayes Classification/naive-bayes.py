import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")

#sınıflandırma için işimize yaraman kısımları çıkarıyoruz 
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

#iyi huylu ve kötü huylu tumorleri ayırıyoruz
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# m değerlerini 1 ve b değerlerini 0 olarak değiştiriryoruz
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#eğitim ve test datalarını oluşturuyoruz
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

#Naive bayes 
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
 
#test
print("naive bayes algoritması doğruluğu: ",nb.score(x_test,y_test))
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 