import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from linkml_runtime.utils.formatutils import underscore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, _tree, export_graphviz

logger = logging.getLogger(__name__)


def tree_to_nested_expression(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    categorical_features: Optional[List[str]] = None,
    feature_encoders: Optional[Dict[str, Union[OneHotEncoder, LabelEncoder]]] = None,
    target_encoder: Optional[LabelEncoder] = None,
) -> str:
    """
    Convert a trained scikit-learn DecisionTreeClassifier to a nested Python conditional expression.

    Args:
    tree (DecisionTreeClassifier): A trained decision tree classifier.
    feature_names (list): List of feature names (including one-hot encoded feature names).
    categorical_features (list): List of original categorical feature names.
    feature_encoders (dict): Dictionary mapping feature names to their respective OneHotEncoders or LabelEncoders.
    target_encoder (LabelEncoder, optional): LabelEncoder for the target variable if it's categorical.

    Returns:
    str: A string representing the nested Python conditional expression.

    Example:
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    >>>
    >>> # Prepare sample data
    >>> X = np.array([[0, 'A'], [0, 'B'], [1, 'A'], [1, 'B']])
    >>> y = np.array(['No', 'Yes', 'Yes', 'No'])
    >>>
    >>> # Prepare the encoders
    >>> feature_encoders = {'feature2': OneHotEncoder(sparse_output=False, handle_unknown='ignore')}
    >>> target_encoder = LabelEncoder()
    >>>
    >>> # Encode the categorical feature and target
    >>> X_encoded = np.column_stack([
    ...     X[:, 0],
    ...     feature_encoders['feature2'].fit_transform(X[:, 1].reshape(-1, 1))
    ... ])
    >>> y_encoded = target_encoder.fit_transform(y)
    >>>
    >>> # Train the decision tree
    >>> clf = DecisionTreeClassifier(random_state=42)
    >>> clf.fit(X_encoded, y_encoded)
    DecisionTreeClassifier(random_state=42)
    >>>
    >>> # Convert to nested expression
    >>> feature_names = ['feature1', 'feature2_A', 'feature2_B']
    >>> categorical_features = ['feature2']
    >>> expression = tree_to_nested_expression(clf, feature_names,
    ...                                        categorical_features, feature_encoders, target_encoder)
    >>> print(expression)
    (("Yes" if ({feature1} <= 0.5000) else "No") if ({feature2} == "A")
       else ("No" if ({feature1} <= 0.5000) else "Yes"))
    """
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    categorical_features = set(categorical_features or [])

    def get_original_feature_name(name):
        return name.split("_")[0] if "_" in name else name

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            original_name = get_original_feature_name(name)
            original_name_safe = underscore(original_name)
            name_safe = underscore(name)

            original_name_safe = "{" + original_name_safe + "}"
            name_safe = "{" + name_safe + "}"

            if original_name in categorical_features:
                if feature_encoders is None or original_name not in feature_encoders:
                    raise ValueError(f"Encoder is required for categorical feature {original_name}")

                encoder = feature_encoders[original_name]
                if isinstance(encoder, OneHotEncoder):
                    # For one-hot encoded features, we check if the specific category is present
                    category = name.split("_", 1)[1]  # Get everything after the first underscore
                    condition = f'{original_name_safe} == "{category}"'
                elif isinstance(encoder, LabelEncoder):
                    category = encoder.inverse_transform([int(threshold)])[0]
                    condition = f'{original_name_safe} == "{category}"'
                else:
                    raise ValueError(f"Unsupported encoder type for feature {original_name}")
            else:
                if np.isinf(threshold):
                    condition = "True"
                else:
                    condition = f"{name_safe} <= {threshold:.4f}"

            left_expr = recurse(tree_.children_left[node])
            right_expr = recurse(tree_.children_right[node])

            return f"({left_expr} if ({condition}) else {right_expr})"
        else:
            class_index = np.argmax(tree_.value[node])
            if target_encoder:
                class_label = target_encoder.inverse_transform([class_index])[0]
                return f'"{class_label}"'
            else:
                return str(class_index)

    return recurse(0)


def escape_label(s: str) -> str:
    """Escape special characters in label strings."""
    s = str(s)
    return re.sub(r"([<>])", r"\\\1", s)


def visualize_decision_tree(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str] = None,
    output_file: Union[Path, str] = "decision_tree.png",
) -> None:
    """
    Generate a visualization of the decision tree and save it as a PNG file.

    :param clf: Trained DecisionTreeClassifier
    :param feature_names: List of feature names
    :param class_names: List of class names (optional)
    :param output_file: The name of the file to save the visualization (default: "decision_tree.png")

    >>> # Create a sample dataset
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'age': [25, 30, 35, 40, 45],
    ...     'income': [50000, 60000, 70000, 80000, 90000],
    ...     'credit_score': [600, 650, 700, 750, 800],
    ...     'approved': ['No', 'No', 'Yes', 'Yes', 'Yes']
    ... })
    >>>
    >>> # Prepare features and target
    >>> X = data[['age', 'income', 'credit_score']]
    >>> y = data['approved']
    >>>
    >>> # Encode target variable
    >>> le = LabelEncoder()
    >>> y_encoded = le.fit_transform(y)
    >>>
    >>> # Train a decision tree
    >>> clf = DecisionTreeClassifier(random_state=42)
    >>> _ = clf.fit(X, y_encoded)
    >>> # Visualize the tree
    >>> visualize_decision_tree(clf, X.columns.tolist(), le.classes_, "tests/output/test_tree.png")
    """
    # Escape special characters in feature names and class names
    escaped_feature_names = [escape_label(name) for name in feature_names]
    escaped_class_names = [escape_label(name) for name in (class_names if class_names is not None else [])]

    import graphviz

    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=escaped_feature_names,
        class_names=escaped_class_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    # dot_data = escape_label(dot_data)
    logger.info(f"Dot: {dot_data}")
    dot_path = shutil.which("dot")
    if not dot_path:
        logger.warning("Graphviz 'dot' executable not found in PATH. Skipping visualization.")
        return
    os.environ["GRAPHVIZ_DOT"] = dot_path

    graph = graphviz.Source(dot_data)
    if isinstance(output_file, Path):
        output_file = str(output_file)
    graph.render(output_file.rsplit(".", 1)[0], format="png", cleanup=True)
