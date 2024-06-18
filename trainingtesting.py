import pandas as pd
df = pd.read_csv('carprices.csv')

import matplotlib.pyplot as plt
plt.scatter(df['Age'], df['Sell Price'])
#plt.show()

X = df[['Mileage', 'Age']]
y = df['Sell Price']
#print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
#print(len(X_train))

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(y_test)

print(clf.score(X_test, y_test))