import glob
import errno
import codecs
import re

import sklearn
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, textblob, string
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

path1 = 'D:/Study/Data/Stylogenetics/সুমন চৌধুরী/*.doc'
path2 = 'D:/Study/Data/Stylogenetics/শুভাশীষ দাশ/*.doc'
path3 = 'D:/Study/Data/Stylogenetics/মুহম্মদ জুবায়ের/*.doc'
path4 = 'D:/Study/Data/Stylogenetics/পুতুল/*.doc'
path5 = 'D:/Study/Data/Stylogenetics/নজরুল ইসলাম/*.doc'

authors, paragraphs = [], []

files = glob.glob(path1)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
            str = " ".join(str.split())
            authors.append(1)
            paragraphs.append(str)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path2)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
            str = " ".join(str.split())
            authors.append(2)
            paragraphs.append(str)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path3)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
            str = " ".join(str.split())
            authors.append(3)
            paragraphs.append(str)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path4)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
            str = " ".join(str.split())
            authors.append(4)
            paragraphs.append(str)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path5)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
            str = " ".join(str.split())
            authors.append(5)
            paragraphs.append(str)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise


traindataframe = pandas.DataFrame()
traindataframe['paragraph'] = paragraphs
traindataframe['author'] = authors

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(traindataframe['paragraph'], traindataframe['author'], test_size=0.4, random_state=45)

author_encoder = preprocessing.LabelEncoder()
train_y = author_encoder.fit_transform(train_y)
valid_y = author_encoder.fit_transform(valid_y)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(traindataframe['paragraph'])
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y)

#Naive Bayes
prediction_accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print ("Naive Bayes: ", prediction_accuracy)

#KNN
prediction_accuracy = train_model(sklearn.neighbors.KNeighborsClassifier(n_neighbors=2), xtrain_count, train_y, xvalid_count)
print ("KNN: ", prediction_accuracy)

#Descision Tree
prediction_accuracy = train_model(sklearn.tree.DecisionTreeClassifier(), xtrain_count, train_y, xvalid_count)
print ("Descision Tree: ", prediction_accuracy)

#SVM
prediction_accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print ("Support Vector Machine: ", prediction_accuracy)

#Logistic Regression
prediction_accuracy = train_model(sklearn.linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print ("Logistic Regression: ", prediction_accuracy)