import pandas as pd

# twitter datasını ekleme
#datamı oku ve içinde latin harfleri var demek 
data = pd.read_csv(r"gender_classifier.csv",encoding = "latin1")
#datanın içerisinden cinsiyet ve textleri alıyrouz 
data = pd.concat([data.gender,data.description],axis=1)
#nan değer içeren satırları çıkarıyoruz 
data.dropna(axis = 0,inplace = True)
#erkekleri 0, kadınları 1 olarak int hale getirdik 
data.gender = [1 if each == "female" else 0 for each in data.gender]

# datayı temizleme  
# regular expression RE mesela "[^a-zA-Z]"
import re

first_description = data.description[4]
# a dan z ye ve A dan Z ye kadar olan harfleri bulma geri kalanları " " (space) ile degistir
description = re.sub("[^a-zA-Z]"," ",first_description)  
# buyuk harftan kucuk harfe cevirme
description = description.lower()   

# stopwords (irrelavent words) gereksiz kelimeler
import nltk # natural language tool kit
nltk.download("stopwords")      # corpus diye bir kalsöre indiriliyor
from nltk.corpus import stopwords  # sonra ben corpus klasorunden import ediyorum

# description = description.split()

# split yerine tokenizer kullanabiliriz
description = nltk.word_tokenize(description)

# split kullanırsak "shouldn't " gibi kelimeler "should" ve "not" diye ikiye ayrılmaz ama word_tokenize() kullanirsak ayrilir

# greksiz kelimeleri cikar
description = [ word for word in description if not word in set(stopwords.words("english"))]
  
             
# lemmatazation loved => love   gitmeyecegim = > git
# kelimelerin köklerini buluyoruz
import nltk as nlp

lemma = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description] 

#son hali ile hepsini metin haline getiriyoruz
description = " ".join(description)

# 4 satır için yaptığımız tüm işlemleri tüm data için sırarı ile uyguluyoruz
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()   # buyuk harftan kucuk harfe cevirme
    description = nltk.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)

# bag of words

# bag of words yaratmak icin kullandigim metot
from sklearn.feature_extraction.text import CountVectorizer 
max_features = 5000

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

#description listimizi count_vectorizer metodu ile fit ederek sparce_matrix e atıyrouz
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x

#en çok kullanılan 500 kelimeyi yazdırıyoruz
print("en sik kullanilan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))

 
y = data.iloc[:,0].values   # erkek ve kadın sınıfı
x = sparce_matrix

# train ve test olmak üzere datayı ayırıyoruz 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


#  naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

# prediction
y_pred = nb.predict(x_test)

print("doğruluk: ",nb.score(y_pred.reshape(-1,1),y_test))



"""
nlp kütüphaneleri :
    nltk
    spacy
    standfor nlp
    open nlp
    
"""






















