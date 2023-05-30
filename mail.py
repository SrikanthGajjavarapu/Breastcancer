#Finding the spam and ham mail using SVC(support vector classifier) model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
data=pd.read_csv('spam.csv',encoding='latin-1')
mail_data=data.where((pd.notnull(data)),' ')
mail_data.head()                      #Spam mail=0 , Ham mail=1
mail_data.loc[mail_data['v1']=='spam','v1',]=0
mail_data.loc[mail_data['v1']=='ham','v1',]=1            
x=mail_data['v2']
y=mail_data['v1']
print(x)
print(...)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
x_train_features=feature_extraction.fit_transform(x_train)
x_test_features=feature_extraction.transform(x_test)
y_train=y_train.astype('int')
y_test=y_test.astype('int')
from sklearn.svm import LinearSVC
model=LinearSVC()
model.fit(x_train_features,y_train)
prediction_on_training_data=model.predict(x_train_features)
accuracy_train=accuracy_score(y_train,prediction_on_training_data)
print(accuracy_train)
test=model.predict(x_test_features)
acurracy_test=accuracy_score(y_test,test)
print(acurracy_test)
input_mail=['WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.']
input_mail_features=feature_extraction.transform(input_mail)
prediction=model.predict(input_mail_features)
print(prediction)
if prediction[0]==1:
  print('HAM MAIL')
else:
  print('SPAM MAIL')