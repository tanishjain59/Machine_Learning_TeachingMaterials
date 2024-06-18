import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target

#print(df[df.target==1])

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
#print(df.head())

from matplotlib import pyplot as plt
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df0 = df[df.target == 2]

# plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color = 'green', marker = '+')
# plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color = 'blue', marker = '.')
# plt.show()

# plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color = 'green', marker = '+')
# plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color = 'blue', marker = '.')
# plt.show()

from sklearn.model_selection import train_test_split
X = df.drop(['target', 'flower_name'], axis = 'columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
from sklearn.svm import SVC
model = SVC(kernel = 'linear')
model.fit(X_train,y_train)
print(model.score(X_test, y_test))
