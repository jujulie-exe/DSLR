
import pandas as pd
from logreg_train import charge_file, cleaning, std_all_input_value, for_loop_gradiend_descent, stdDeV, PATH_DATA_SET
import sys
from matplotlib import pyplot as plt

def histogram(data: pd.DataFrame, houses: pd.Series) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))
    for i, column in enumerate(data.columns):
        values = data[column].groupby(houses).apply(list).to_dict()
        for house, vals in values.items():
            axes[i//4, i%4].hist(vals, bins=10, alpha=0.5, label=house)
        axes[i//4, i%4].set_title(f"Histogram of {column} by Hogwarts House")
        axes[i//4, i%4].set_xlabel(column)
        axes[i//4, i%4].set_ylabel("Frequency")
        axes[i//4, i%4].legend()
    plt.tight_layout()
    plt.show()



def main() -> int:
    try:
        data_raw: pd.DataFrame = charge_file(PATH_DATA_SET)
        data_cleaning, houses, materie = cleaning(data_raw)
        # print("house: ", houses)
        # print("materie: ", materie)
        # print("data_cleaning: ", data_cleaning)
        data_std = std_all_input_value(data_cleaning)
        histogram(data_std, houses)
        plt.show()
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