### THIS IS SINGLE CLASS CLASSIFICATION

"""import pandas as pd
df = pd.read_csv('insurance_data.csv')

X = df[['Age']]
y = df['Insurance']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# print(model.predict(X_test))
# print(y_test)
# print(model.score(X_test, y_test))

print(model.predict_proba(X_test))"""

### MULTI-CLASS CLASSIFICATION

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
plt.gray()
#plt.matshow(digits.images[0])
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

#print(model.predict([digits.data[67]]))

y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()