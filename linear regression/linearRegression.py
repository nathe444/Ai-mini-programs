import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data = {
    'area': [2600, 3000, 3200, 3600, 4000],
    'price': [550000, 565000, 610000, 680000, 725000]
}
df = pd.DataFrame(data)

plt.scatter(df.area , df.price)
plt.xlabel("area")
plt.ylabel("price")
plt.title("Prediction")


reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])

predicted_price = (reg.predict([[3222],[8900],[2890]]))

print(predicted_price)

plt.plot(df.area, reg.predict(df[['area']]), color='red')


plt.show()
