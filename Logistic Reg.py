import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv(r"C:\Users\ahmed\Documents\Machine Learning\assignment1\brca.csv")
df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})
corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

X = df.loc[:, df.columns != 'diagnosis']
Y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)
print(y_pred)

confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()


accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
