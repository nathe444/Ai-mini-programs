from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score

digits = load_digits()

k_fold = KFold(n_splits=4, shuffle=False , random_state=None)

for train_index, test_index in k_fold.split(digits.data):
  x_train ,x_test ,y_train ,y_test = digits.data[train_index] ,digits.data[test_index] ,digits.target[train_index] ,digits.target[test_index]

print('Logistic Regression scores:' , cross_val_score(LogisticRegression(max_iter=10000), x_train, y_train, cv=k_fold, scoring='accuracy'))

print('SVC scores:' , cross_val_score(SVC(), x_train, y_train, cv=k_fold, scoring='accuracy'))

print('Random Forest scores:' , cross_val_score(RandomForestClassifier(n_estimators=50), x_train, y_train, cv=k_fold, scoring='accuracy'))