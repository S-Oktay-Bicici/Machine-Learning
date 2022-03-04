# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sklearn library
from sklearn.linear_model import LinearRegression

# import data
#csv formatını ; ile dataframe formatına çevirerek datayı alıyoruz 
df = pd.read_csv("linear-regression-dataset.csv",sep = ";")

# plot data
#aldığımız datayı görselleştiriyoruz
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()


# linear regression model
linear_reg = LinearRegression()

#arraye shape ile bakıldığına (14,) olan durumu (14,1) yapmak için resahape yapıyoruz
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

# prediction

b0 = linear_reg.predict(0)
print("b0: ",b0)

b0_ = linear_reg.intercept_
print("b0_: ",b0_)   # y eksenini kestigi nokta intercept

b1 = linear_reg.coef_
print("b1: ",b1)   # egim slope

# maas = 1663 + 1138*deneyim 

maas_yeni = 1663 + 1138*11
print(maas_yeni)

print(linear_reg.predict(11))

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # deneyim

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)  # maas

plt.plot(array, y_head,color = "red")

linear_reg.predict(100)










