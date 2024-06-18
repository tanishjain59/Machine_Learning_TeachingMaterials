import pandas as pd
df = pd.read_csv('spam.csv')
df.groupby('Category')

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam,test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)

emails = [
    'Hey tanish, can we get together to watch the free game tomorrow?',
    'Upto 20% on parking, exclusive offering just for Tanish!'
]
emails_count = v.transform(emails)
#print(model.predict(emails_count))

X_test_count = v.transform(X_test)
print(model.score(X_test_count,y_test))
print(model.predict_proba(X_test_count[:10]))

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())    
])
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))