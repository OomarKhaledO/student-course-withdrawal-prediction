from preprocessing import load_and_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

k_values = [1, 3, 5, 7, 9, 11]

best_k = 0
best_score = 0


print("\n===== CROSS VALIDATION =====\n")

for k in k_values:

    scores = []

    for train_index, val_index in skf.split(X_train, y_train):

        X_tr = X_train[train_index]
        X_val = X_train[val_index]

        y_tr = y_train.iloc[train_index]
        y_val = y_train.iloc[val_index]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr, y_tr)

        pred = knn.predict(X_val)

        scores.append(f1_score(y_val, pred))

    avg = np.mean(scores)

    print(f"K = {k} | F1 = {avg}")

    if avg > best_score:
        best_score = avg
        best_k = k


print("\nBEST K:", best_k)
print("BEST SCORE:", best_score)

print("\n===== FINAL TEST =====\n")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))