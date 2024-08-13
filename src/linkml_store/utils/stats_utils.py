import numpy as np
import pandas as pd


def predictive_power(df, target_col, feature_cols, cv=5):
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier

    # Prepare the data
    X = df[feature_cols].copy()  # Create an explicit copy
    y = df[target_col].copy()

    # Encode categorical variables
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    # Adjust cv based on the number of unique values in y
    n_unique = len(np.unique(y))
    cv = min(cv, n_unique)

    # Train a decision tree and get cross-validated accuracy
    clf = DecisionTreeClassifier(random_state=42)

    if cv < 2:
        # If cv is less than 2, we can't do cross-validation, so we'll just fit and score
        clf.fit(X, y)
        return clf.score(X, y)
    else:
        scores = cross_val_score(clf, X, y, cv=cv)
        return scores.mean()


def analyze_predictive_power(df, columns=None, cv=5):
    if columns is None:
        columns = df.columns
    results = pd.DataFrame(index=columns, columns=["predictive_power", "features"])

    for target_col in columns:
        feature_cols = [col for col in columns if col != target_col]
        try:
            power = predictive_power(df, target_col, feature_cols, cv)
            results.loc[target_col, "predictive_power"] = power
            results.loc[target_col, "features"] = ", ".join(feature_cols)
        except Exception as e:
            print(f"Error processing {target_col}: {str(e)}")
            results.loc[target_col, "predictive_power"] = np.nan

    return results
