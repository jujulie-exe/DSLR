
import pandas as pd
from logreg_train import charge_file, cleaning, std_all_input_value, for_loop_gradiend_descent, stdDeV, PATH_DATA_SET
import sys
from matplotlib import pyplot as plt
import seaborn as sns
def plot_one(data: pd.DataFrame, page_i: int, page_j: int, columns: list, houses: list) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(8, 4*len(columns)))
    colors = sns.color_palette("hls", len(houses))
    color_map = {house: color for house, color in zip(houses, colors)}

    houses_encoded = pd.Categorical(houses).codes

    for i in range(page_i * 5, min((page_i + 1) * 5, len(columns))):
        for j in range(page_j * 5, min((page_j + 1) * 5, len(columns))):
            value_x = i % 5
            value_y = j % 5
            if value_x == 0:
                axes[value_x, value_y].set_title(columns[j])
            if value_y == 0:
                axes[value_x, value_y].set_ylabel(columns[i])
            if i == j:
                axes[value_x, value_y].hist(data[columns[i]], bins=10, alpha=0.5)
            else:
                axes[value_x, value_y].scatter(data[columns[j]], data[columns[i]], alpha=0.5, c=houses_encoded, cmap='tab10')
    plt.show()

    

def pair_plot(data: pd.DataFrame, houses: list) -> None:
    columns = data.columns.tolist()
    plot_one(data, 0, 0, columns, houses)
    plot_one(data, 0, 1, columns, houses)
    # plot_one(data, 1, 0, columns, houses)
    plot_one(data, 1, 1, columns, houses)

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
        # scatter plot
        data_std = data_std.drop(columns=["Defense Against the Dark Arts"])


        pair_plot(data_std, houses)
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