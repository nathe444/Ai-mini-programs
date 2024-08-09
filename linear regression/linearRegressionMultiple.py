import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = {
  'area': [2600, 3000, 3200, 3600, 4000],
  'bedrooms': [3, 4, 3, 5, 5],
  'age':[20,15,18,30,8],
  'price':[550000, 565000, 610000, 595000, 760000]
}



df = pd.DataFrame(data)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df['price'])

predicted = reg.predict([[3000,4,15],[3400,2,19]])

print(predicted)

