import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess():

    # Load dataset
    df = pd.read_csv("./Dataset/WithdrawlStudents.csv")


    # Missing values - numeric
    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())


    # Missing values - categorical
    categorical_columns = df.select_dtypes(include="object").columns

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])


    # Encoding (FIXED: separate encoder per column)
    encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le


    # Split features/target
    X = df.drop(columns=["withdrawl", "final-result"])
    y = df["withdrawl"]


    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Cross validation object
    skf = StratifiedKFold(
        n_splits=7,
        shuffle=True,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, skf, scaler, encoders


# ONLY runs when file is executed directly
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()
    print("Preprocessing completed successfully.")