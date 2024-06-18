from sklearn import svm, datasets
iris = datasets.load_iris()

import pandas as pd
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
model = svm.SVC(kernel = 'rbf', C=30, gamma = 'auto')
model.fit(X_train, y_train)
print(model.score(X_test,y_test))

import numpy as np
from sklearn.model_selection import cross_val_score
kernels = ['rbf', 'linear']
C=[1,10,20]
avg_scores={}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel = kval, C = cval, gamma = 'auto'), iris.data, iris.target, cv=5)
        avg_scores[kval+'_'+str(cval)] = np.average(cv_scores)
print(max(avg_scores))

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C':[1,10,20],
    'kernel':['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
df = pd.DataFrame(clf.cv_results_)

print(df[['param_C', 'param_kernel', 'mean_test_score']])
print(clf.best_params_)

from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
    'C':[1,10,20],
    'kernel':['rbf','linear']
}, cv=5, return_train_score=False, n_iter=2)
rs.fit(iris.data, iris.target)
df2 = pd.DataFrame(rs.cv_results_)[['param_C', 'param_kernel', 'mean_test_score']]
print(df2)

###HOW TO CHOOSE BASE MODEL

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C':[1,10,20],
            'kernel': ['rbf', 'linear']
        }

    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators':[1,5,10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'C':[1,5,10]
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })


df3 = pd.DataFrame(scores, columns=['model','best_score','best_params'])
print(df3)