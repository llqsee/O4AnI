import pandas as pd
import numpy as np

def sort_dataframe(df, attribute=None, order=None, ascending=True):
    """
    Sorts a DataFrame based on an attribute.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to sort.
    - attribute (str, optional): The column name to sort by. If None, shuffles the DataFrame.
    - order (list, optional): The custom order for ordinal data.
    - ascending (bool, optional): Whether to sort numerically in ascending order (default: True).
    
    Returns:
    - pd.DataFrame: The sorted or shuffled DataFrame.
    """
    if attribute is None:
        # Shuffle the DataFrame randomly
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if attribute not in df.columns:
        raise ValueError(f"Attribute '{attribute}' not found in DataFrame.")
    
    dtype = df[attribute].dtype

    if np.issubdtype(dtype, np.number):  
        # Numeric sorting
        return df.sort_values(by=attribute, ascending=ascending)

    elif order is not None and isinstance(order, list):  
        # Ordinal sorting
        if not set(df[attribute].unique()).issubset(set(order)):
            raise ValueError("Order list must include all unique values in the column.")

        return df.assign(
            **{attribute: pd.Categorical(df[attribute], categories=order, ordered=True)}
        ).sort_values(by=attribute)

    else:  
        # Random sorting for non-numeric, non-ordinal attributes
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

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
