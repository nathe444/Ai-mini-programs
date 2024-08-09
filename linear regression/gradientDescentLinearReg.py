import numpy as np

def gradientDescent(x,y):
  m_current = b_current = 0
  iterations =10
  learning_rate = 0.001
 
  n = len(x)

  for i in range(iterations):
    y_predicted = m_current * x + b_current
    md = -2/n * np.sum(x * (y-y_predicted))
    bd = -2/n * np.sum(y-y_predicted)
    
    m_current = m_current- learning_rate* md
    b_current = b_current-learning_rate*bd

    cost = 1/n * np.sum((y-y_predicted)**2)

    print(f"m = {m_current} b={b_current} iterations = {i} cost ={cost}")

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
gradientDescent(x, y)
