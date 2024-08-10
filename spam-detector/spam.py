import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


df = pd.read_csv('spam.csv')
df['spam']= df['Category'].apply(lambda x: 1 if x=='spam' else 0)

x_train,x_test,y_train,y_test = train_test_split(df.Message,df.spam,test_size=0.2)

# count = CountVectorizer()
# x_train_count = count.fit_transform(x_train.values)

# model = MultinomialNB()
# model.fit(x_train_count,y_train)


spam_detector = Pipeline([
    ('vectorizer', CountVectorizer()),   
    ('classifier', MultinomialNB())      
])


spam_detector.fit(x_train,y_train)

print(spam_detector.score(x_test,y_test))




