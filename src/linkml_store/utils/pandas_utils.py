from typing import Dict, List, Tuple, Union

import pandas as pd


def facet_summary_to_dataframe_unmelted(
    facet_summary: Dict[Union[str, Tuple[str, ...]], List[Tuple[Union[str, Tuple[str, ...]], int]]]
) -> pd.DataFrame:
    rows = []

    for facet_type, facet_data in facet_summary.items():
        if isinstance(facet_type, str):
            # Single facet type
            for category, value in facet_data:
                rows.append({facet_type: category, "Value": value})
        else:
            # Multiple facet types
            for cat_val_tuple in facet_data:
                if len(cat_val_tuple) == 2:
                    categories, value = cat_val_tuple
                else:
                    categories, value = cat_val_tuple[:-1], cat_val_tuple[-1]
                row = {"Value": value}
                for i, facet in enumerate(facet_type):
                    row[facet] = categories[i]
                rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all columns are present, fill with None if missing
    all_columns = set(col for facet in facet_summary.keys() for col in (facet if isinstance(facet, tuple) else [facet]))
    for col in all_columns:
        if col not in df.columns:
            df[col] = None

    # Move 'Value' to the end
    cols = [col for col in df.columns if col != "Value"] + ["Value"]
    df = df[cols]

    return df
