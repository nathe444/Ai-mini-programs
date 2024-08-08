import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('csv/titanic.csv')

inputs = df.drop(['PassengerId' , 'Name' ,'SibSp','Parch','Ticket','Cabin','Embarked','Survived'],axis='columns')

target = df['Survived']


le_age = LabelEncoder()
le_fare = LabelEncoder()
le_pclass = LabelEncoder()
le_sex= LabelEncoder()

inputs['Age_n'] = le_age.fit_transform(inputs['Age'])
inputs['Fare_n'] = le_age.fit_transform(inputs['Fare'])
inputs['Pclass_n'] = le_pclass.fit_transform(inputs['Pclass'])
inputs['Sex_n'] = le_sex.fit_transform(inputs['Sex'])

inputs.drop(['Age','Fare','Pclass','Sex'],axis='columns',inplace=True)

model = tree.DecisionTreeClassifier()
model_1 = RandomForestClassifier(n_estimators=40)


x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

model.fit(x_train,y_train)

model_1.fit(x_train,y_train)

print(model.score(x_test,y_test))

print(model_1.score(x_test,y_test))






