from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(X_train,y_train)