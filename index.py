import pandas as pd

dataset = pd.read_csv("Breast_Cancer.csv")

dataset.head()

dataset.info()

X = dataset.drop(columns=["Status"])
y = dataset.Status

print(X)
print("-----------------")
print(y)

dataset.Status.value_counts()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_ix = X.select_dtypes(include=['object']).columns
t = [('cat', OneHotEncoder(),
      categorical_ix)]
col_transform = ColumnTransformer(transformers=t)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn import tree

dt = tree.DecisionTreeClassifier(random_state=1, max_depth=9)

pipeline = Pipeline(steps=[('prep',col_transform),
                           ('model',dt)])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("%.2f" % accuracy)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

pipeline_knn = Pipeline(steps=[('prep',col_transform),
                           ('model',knn)])

pipeline_knn.fit(X_train, y_train)

y_pred_knn = pipeline_knn.predict(X_test)
acc_knn = accuracy_score(y_test,y_pred_knn)
print("%.2f" % acc_knn)

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer

nb = GaussianNB()

func_tranf = FunctionTransformer(lambda x: x.todense(), accept_sparse=True)

pipeline_nb = Pipeline(steps=[('prep',col_transform),
                              ('func',func_tranf),
                              ('model',nb)])

pipeline_nb.fit(X_train, y_train)

y_pred_nb = pipeline_nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
print("%.2f" % acc_nb)
