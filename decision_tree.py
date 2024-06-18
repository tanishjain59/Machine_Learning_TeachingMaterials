import pandas as pd 
df = pd.read_csv('salaries.csv')

#print(df)

targets = df['salary_more_then_100k']
inputs = df.drop('salary_more_then_100k', axis = 'columns')

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_company.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])

inputs_n = inputs.drop(['company', 'job', 'degree'], axis = 'columns')
#print(inputs_n)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, targets)
print(model.score(inputs_n, targets))