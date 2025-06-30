import pandas as pd
import numpy as np
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_categorical_dtype,
)

def sort_dataframe(df: pd.DataFrame,
                   attribute: str = None,
                   order: list = None,
                   ascending: bool = True) -> pd.DataFrame:
    """
    Sort a DataFrame by `attribute`.  
      - If attribute is None: reproducibly shuffle.  
      - If numeric or datetime: natural sort.  
      - If `order` is given for any type: treat column as ordered Categorical.  
      - If categorical without order: natural sort by category codes.  
      - Otherwise: shuffle.
    """
    # 1) Shuffle if no attribute
    if attribute is None:
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2) Ensure column exists
    if attribute not in df.columns:
        raise ValueError(f"Attribute '{attribute}' not found in DataFrame.")

    col = df[attribute]

    # 3) If order list provided → ordinal sort
    if order is not None:
        missing = set(col.unique()) - set(order)
        if missing:
            raise ValueError(f"Order list is missing values: {missing}")
        cat_type = pd.CategoricalDtype(categories=order, ordered=True)
        df2 = df.copy()
        df2[attribute] = df2[attribute].astype(cat_type)
        return df2.sort_values(by=attribute, ascending=ascending).reset_index(drop=True)

    # 4) Numeric
    if is_numeric_dtype(col):
        return df.sort_values(by=attribute, ascending=ascending).reset_index(drop=True)

    # 5) Datetime
    if is_datetime64_any_dtype(col):
        return df.sort_values(by=attribute, ascending=ascending).reset_index(drop=True)

    # 6) Already Categorical (but no order) → sort by codes
    if is_categorical_dtype(col):
        return df.sort_values(by=attribute, ascending=ascending).reset_index(drop=True)

    # 7) Fallback: shuffle
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# def sort_dataframe(df, attribute=None, order=None, ascending=True):
#     """
#     Sorts a DataFrame based on an attribute.
    
#     Parameters:
#     - df (pd.DataFrame): The DataFrame to sort.
#     - attribute (str, optional): The column name to sort by. If None, shuffles the DataFrame.
#     - order (list, optional): The custom order for ordinal data.
#     - ascending (bool, optional): Whether to sort numerically in ascending order (default: True).
    
#     Returns:
#     - pd.DataFrame: The sorted or shuffled DataFrame.
#     """
#     if attribute is None:
#         # Shuffle the DataFrame randomly
#         return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
#     if attribute not in df.columns:
#         raise ValueError(f"Attribute '{attribute}' not found in DataFrame.")
    
#     dtype = df[attribute].dtype

#     if np.issubdtype(dtype, np.number):  
#         # Numeric sorting
#         return df.sort_values(by=attribute, ascending=ascending)

#     elif order is not None and isinstance(order, list):  
#         # Ordinal sorting
#         if not set(df[attribute].unique()).issubset(set(order)):
#             raise ValueError("Order list must include all unique values in the column.")

#         return df.assign(
#             **{attribute: pd.Categorical(df[attribute], categories=order, ordered=True)}
#         ).sort_values(by=attribute)

#     else:  
#         # Random sorting for non-numeric, non-ordinal attributes
#         return df.sample(frac=1, random_state=42).reset_index(drop=True)



# Example usage
if __name__ == "__main__":
    data = {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 22, 35],
        "level": ["junior", "senior", "mid", "senior"]
    }
    
    df = pd.DataFrame(data)
    
    # Sorting numerically
    print("Sorted by age (ascending):")
    print(sort_dataframe(df, "age", ascending=True))

    # Sorting ordinally
    print("\nSorted by level (ordinal order):")
    print(sort_dataframe(df, "level", order=["junior", "mid", "senior"]))

    # Random sorting
    print("\nRandomly sorted:")
    print(sort_dataframe(df))
