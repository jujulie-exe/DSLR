# recreate comand  pandas.DataFrame.describe 
import pandas as pd
from logreg_train import charge_file, cleaning, std_all_input_value, for_loop_gradiend_descent, stdDeV, PATH_DATA_SET
import sys


def count(column: pd.Series) -> int:
    count = 0
    for value in column:
        if pd.notna(value):
            count += 1
    return count


def mean(column: pd.Series) -> float:
    ret: float = column.sum() / count(column)
    if pd.isna(ret):
        return float('nan')
    return ret

def std(column: pd.Series) -> float:
    mean_value = mean(column)
    variance = sum((value - mean_value) ** 2 for value in column if pd.notna(value)) / (count(column) - 1)
    return variance ** 0.5


def quantile(column: pd.Series, q: float) -> float:
    values = [value for value in column if pd.notna(value)]
    values.sort()
    n = len(values)
    if n == 0:
        return float('nan')
    
    index: float = q * (n - 1)
    lower_index = int(index)
    upper_index = lower_index + 1

    if upper_index > n - 1:
        upper_index = n - 1

    weight = index - lower_index
    return (1 - weight) * values[lower_index] + weight * values[upper_index]

def min(column: pd.Series) -> float:
    min_value = float('inf')
    for value in column:
        if pd.notna(value) and value < min_value:
            min_value = value
    return min_value

def max(column: pd.Series) -> float:
    max_value = float('-inf')
    for value in column:
        if pd.notna(value) and value > max_value:
            max_value = value
    return max_value

def diff(value1: float, value2: float) -> None:
    value1 = round(value1, 6)
    value2 = round(value2, 6)
    if (value1 == value2):
        print(value1)
    else:
        print(f"Difference found: {value1} != {value2}")
    

def describe(data_raw: pd.DataFrame, percentiles: list = [0.25, 0.5, 0.75]) -> None:
    for column in data_raw.columns:
        if pd.api.types.is_numeric_dtype(data_raw[column]):
            print(f"Column: {column}")
            print("Count:" , end=" ")
            diff(data_raw[column].count(), count(data_raw[column]))
            print(f"Mean:", end=" ")
            diff(data_raw[column].mean(), mean(data_raw[column]))
            print(f"Std:", end=" ")
            diff(data_raw[column].std(), std(data_raw[column]))
            print(f"Min:", end=" ")
            diff(data_raw[column].min(), min(data_raw[column]))
            for p in percentiles:
                print(f"{p*100}%:", end=" ")
                diff(data_raw[column].quantile(p), quantile(data_raw[column], p))
            print(f"Max:", end=" ")
            diff(data_raw[column].max(), max(data_raw[column]))
            print("----------------------------------------")




def main() -> int:
    try:
        data_raw: pd.DataFrame = charge_file(PATH_DATA_SET)
        print(data_raw.describe(percentiles=[0.25, 0.5, 0.75], include='all'))
        describe(data_raw)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        return 1
    except pd.errors.ParserError:
        print("Error: The file is corrupt")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
