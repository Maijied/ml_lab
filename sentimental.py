 import pandas
 import pandas as pd
 import sklearn
 from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, tree, neighbors
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

 df = pd.read_csv('D:/Study/Data/Sentimental Analysis/sentiment.csv')
 datas,sentiments=[],[]
 for index, row in df.iterrows():
     rowstring=row['data;title'].split(';')
     if(rowstring[1]=='Like (ভাল)'):
         datas.append(rowstring[0])
         sentiments.append(0)
     if(rowstring[1]=='Smiley (স্মাইলি)'):
         datas.append(rowstring[0])
         sentiments.append(1)
     if(rowstring[1]=='HaHa(হা হা)'):
         datas.append(rowstring[0])
         sentiments.append(2)
     if(rowstring[1] == 'Sad (দু: খিত)'):
         datas.append(rowstring[0])
         sentiments.append(3)
     if(rowstring[1]=='Skip ( বোঝতে পারছি না )'):
         datas.append(rowstring[0])
         sentiments.append(4)
     if(rowstring[1]=='Love(ভালবাসা)'):
         datas.append(rowstring[0])
         sentiments.append(5)
     if(rowstring[1]=='WOW(কি দারুন)'):
         datas.append(rowstring[0])
         sentiments.append(6)
     if(rowstring[1]=='Blush(গোলাপী আভা)'):
         datas.append(rowstring[0])
         sentiments.append(7)
     if(rowstring[1]=='Consciousness (চেতনাবাদ)'):
         datas.append(rowstring[0])
         sentiments.append(8)
     if(rowstring[1]=='Rocking (আন্দোলিত হত্তয়া)'):
         datas.append(rowstring[0])
         sentiments.append(9)
     if(rowstring[1]=='Bad (খারাপ)'):
         datas.append(rowstring[0])
         sentiments.append(10)
     if(rowstring[1]=='Angry (রাগান্বিত)'):
         datas.append(rowstring[0])
         sentiments.append(11)
     if(rowstring[1]=='Fail (ব্যর্থ)'):
         datas.append(rowstring[0])
         sentiments.append(12)
     if(rowstring[1]=='Provocative (উস্কানিমুলক)'):
         datas.append(rowstring[0])
         sentiments.append(13)
     if(rowstring[1]=='Shocking (অতিশয় বেদনাদায়ক)'):
         datas.append(rowstring[0])
         sentiments.append(14)
     if(rowstring[1]=='Protestant (প্রতিবাদমূলক)'):
         datas.append(rowstring[0])
         sentiments.append(15)
     if(rowstring[1]=='Evil (জঘন্য)'):
         datas.append(rowstring[0])
         sentiments.append(16)
     if(rowstring[1]=='Skeptical (সন্দেহপ্রবণ)'):
         datas.append(rowstring[0])
         sentiments.append(17)


 traindataframe = pandas.DataFrame()
 traindataframe['data'] = datas
 traindataframe['sentiment'] = sentiments
 train_x, valid_x, train_y, valid_y = model_selection.train_test_split(traindataframe['data'], traindataframe['sentiment'], test_size=0.2,random_state=40)
 data_encoder = preprocessing.LabelEncoder()
 train_y = data_encoder.fit_transform(train_y)
 valid_y = data_encoder.fit_transform(valid_y)
 count_vector = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
 count_vector.fit(traindataframe['data'])
 xtrain_count = count_vector.transform(train_x)
 xvalid_count = count_vector.transform(valid_x)

 #Naive Bayes
model=naive_bayes.MultinomialNB()
model.fit(xtrain_count,train_y)
sentiment_prediction=model.predict(xvalid_count)
print("Naive Bayes: ",metrics.accuracy_score(sentiment_prediction, valid_y))

# #Descision Tree
model= sklearn.tree.DecisionTreeClassifier()
model.fit(xtrain_count,train_y)
sentiment_prediction=model.predict(xvalid_count)
print("Descision Tree: ",metrics.accuracy_score(sentiment_prediction, valid_y))

# #K-Nearest Neighbour
model=sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain_count,train_y)
sentiment_prediction=model.predict(xvalid_count)
print("KNN: ",metrics.accuracy_score(sentiment_prediction, valid_y))

# #Support Vector Machine
model= svm.SVC()
model.fit(xtrain_count,train_y)
sentiment_prediction=model.predict(xvalid_count)
print("SVM: ",metrics.accuracy_score(sentiment_prediction, valid_y))

# #Logistic Regression
model= sklearn.linear_model.LogisticRegression()
model.fit(xtrain_count,train_y)
sentiment_prediction=model.predict(xvalid_count)
print("Logistic Regression: ",metrics.accuracy_score(sentiment_prediction, valid_y))