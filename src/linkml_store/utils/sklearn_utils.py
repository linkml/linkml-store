import numpy as np


def tree_to_nested_expression(tree, feature_names, categorical_features=None, feature_encoders=None,
                              target_encoder=None):
    """
    Convert a trained scikit-learn DecisionTreeClassifier to a nested Python conditional expression.

    Args:
    tree (DecisionTreeClassifier): A trained decision tree classifier.
    feature_names (list): List of feature names.
    categorical_features (list): List of categorical feature names.
    feature_encoders (dict): Dictionary mapping feature names to their respective LabelEncoders.
    target_encoder (LabelEncoder, optional): LabelEncoder for the target variable if it's categorical.

    Returns:
    str: A string representing the nested Python conditional expression.

    Example:
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.preprocessing import LabelEncoder
    >>>
    >>> # Prepare sample data
    >>> X = np.array([[0, 'A'], [0, 'B'], [1, 'A'], [1, 'B']])
    >>> y = np.array(['No', 'Yes', 'Yes', 'No'])
    >>>
    >>> # Prepare the encoders
    >>> feature_encoders = {'feature2': LabelEncoder()}
    >>> target_encoder = LabelEncoder()
    >>>
    >>> # Encode the categorical feature and target
    >>> X_encoded = X.copy()
    >>> X_encoded[:, 1] = feature_encoders['feature2'].fit_transform(X[:, 1])
    >>> y_encoded = target_encoder.fit_transform(y)
    >>>
    >>> # Train the decision tree
    >>> clf = DecisionTreeClassifier(random_state=42)
    >>> clf.fit(X_encoded, y_encoded)
    DecisionTreeClassifier(random_state=42)
    >>> # Convert to nested expression
    >>> feature_names = ['feature1', 'feature2']
    >>> categorical_features = ['feature2']
    >>> expression = tree_to_nested_expression(clf, feature_names, categorical_features, feature_encoders, target_encoder)
    >>> print(expression)
    (("No" if (feature2 == "A") else "Yes") if (feature1 <= 0.5000) else ("Yes" if (feature2 == "A") else "No"))
    """
    from sklearn.tree import _tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    categorical_features = set(categorical_features or [])

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name in categorical_features:
                if feature_encoders is None or name not in feature_encoders:
                    raise ValueError(f"Encoder is required for categorical feature {name}")
                category = feature_encoders[name].inverse_transform([int(threshold)])[0]
                condition = f'{name} == "{category}"'
            else:
                condition = f'{name} <= {threshold:.4f}'

            left_expr = recurse(tree_.children_left[node])
            right_expr = recurse(tree_.children_right[node])

            return f'({left_expr} if ({condition}) else {right_expr})'
        else:
            class_index = np.argmax(tree_.value[node])
            if target_encoder:
                class_label = target_encoder.inverse_transform([class_index])[0]
                return f'"{class_label}"'
            else:
                return str(class_index)

    return recurse(0)
