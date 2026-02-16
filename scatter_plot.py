import pandas as pd
from logreg_train import charge_file, cleaning, std_all_input_value, for_loop_gradiend_descent, stdDeV, PATH_DATA_SET
from describe import mean
import sys
from matplotlib import axes, axes, pyplot as plt

def covariance(x: pd.Series, y: pd.Series) -> float:
    mean_x = mean(x)
    mean_y = mean(y)
    cov = mean((x - mean_x) * (y - mean_y))
    return cov

def scatter_plot(data: pd.DataFrame) -> None:
    columns = data.columns.tolist()
    max_covariance = float('-inf')
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            x = data[columns[i]]
            y = data[columns[j]]
            cov = covariance(x, y)
            if abs(cov) > max_covariance:
                max_covariance = abs(cov)                
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.scatter(x, y, alpha=0.5)
                ax.set_title(f"Scatter Plot of {columns[i]} vs {columns[j]}\nCovariance: {cov:.2f}")
                ax.set_xlabel(columns[i])
                ax.set_ylabel(columns[j])
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

        # histogram
        data_std = data_std.drop(columns=["Arithmancy"])
        data_std = data_std.drop(columns=["Care of Magical Creatures"])
        data_std = data_std.drop(columns=["Bias"])

        scatter_plot(data_std)
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