
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = {
    'Age': [22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 23, 27, 29, 34, 38, 42, 47, 52, 57, 62, 67, 72, 77, 28, 33, 37, 43],
    'Insurance': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

x_train, x_test, y_train, y_test = train_test_split(df[['Age']], df.Insurance, test_size=0.1, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

predictions = model.predict([[10], [19], [29], [39]])
print("Predictions for ages 10, 19, 29, 39:", predictions)

plt.title("Age vs. Insurance")
plt.ylabel("Insurance")
plt.xlabel("Age")
plt.scatter(df.Age, df.Insurance, color='blue', label='Actual data')

# Generating a sequence of ages to plot the prediction probabilities
ages = np.linspace(20, 80, 300).reshape(-1, 1)
probabilities = model.predict_proba(ages)[:, 1]

# Plotting the sigmoid (logistic regression) curve
plt.plot(ages, probabilities, color='red', label='Sigmoid curve')
plt.legend()
plt.show()
