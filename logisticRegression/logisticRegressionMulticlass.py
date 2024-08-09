import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(x_train, y_train)


print("Model Accuracy:", model.score(x_test, y_test))

predicted_digit = model.predict([digits.data[89]])
print(predicted_digit)

plt.gray()
plt.matshow(digits.images[30])
plt.title(f"Predicted: {predicted_digit[0]}")
plt.show()
