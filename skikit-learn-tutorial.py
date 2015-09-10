__author__ = 'maxi'
from sklearn import datasets
from matplotlib import pyplot as plt


#iris = datasets.load_iris()
#digits = datasets.load_digits()
house_prices = datasets.load_boston()
print(house_prices.data)
#print(digits.data)
plt.plot(house_prices.data)
plt.show()