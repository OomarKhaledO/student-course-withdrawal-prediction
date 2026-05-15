from preprocessing import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

C_values = [0.01, 0.1, 1, 10, 100]

best_C = 0
best_score = 0


print("\n===== CROSS VALIDATION =====\n")

for C in C_values:

    scores = []

    for train_index, val_index in skf.split(X_train, y_train):

        X_tr = X_train[train_index]
        X_val = X_train[val_index]

        y_tr = y_train.iloc[train_index]
        y_val = y_train.iloc[val_index]

        lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
        lr.fit(X_tr, y_tr)

        pred = lr.predict(X_val)

        scores.append(f1_score(y_val, pred))

    avg = np.mean(scores)

    print(f"C = {C} | F1 = {avg}")

    if avg > best_score:
        best_score = avg
        best_C = C


print("\nBEST C:", best_C)
print("BEST SCORE:", best_score)

print("\n===== FINAL TEST =====\n")

lr = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))