import logreg_train as lrt
import pandas as pd
import numpy as np
import numpy.typing as npt
import sys
import os
"""
House,Arithmancy,Astronomy,Herbology,Defense Against the Dark Arts,Divination,Muggle Studies,Ancient Runes,History of Magic,Transfiguration,Potions,Care of Magical Creatures,Charms,Flying,Bias
Ravenclaw,0.015123986565551065,-0.3469728246909958,0.25581485023916695,0.34849404806419415,0.212222514129826,0.571320775701334,0.45218520883689023,0.07106546147643868,0.07921034318608203,0.05188670718067233,0.015566872801775234,0.521500947672387,-0.00453987340142728,-0.5117367147648437
Slytherin,-0.005276704775866907,-0.35022112283714363,-0.47857763830352174,0.3497420501213186,-0.7269862111329293,-0.22574144275139596,-0.29783356310057024,0.05006655839194799,0.09431806889458637,0.3338726351680526,-0.008103191419982957,-0.34268337272852667,-0.2614123409985443,-1.0145282717704178
Gryffindor,-0.03894437676613778,0.13844653358189749,-0.4231534626425835,-0.13663950260797592,0.11063935812576124,-0.08251884885512548,0.39213489643442045,-0.5119278983559782,-0.5481311017476188,-0.1895303007957854,-0.027083807214351534,-0.26521373495752015,0.5036205230870024,-0.8326057662089026
Hufflepuff,0.023623906715511837,0.41016407065984817,0.37420853640997276,-0.41198301780473007,0.21711114711027268,-0.25767934980079066,-0.43229606928643827,0.2438398039637122,0.23721187141063527,-0.13512788487832222,0.010507352121504195,-0.030235390740982217,-0.16356103362008328,-0.3194610133013671
"""
def write_int_test_file(predicted_houses: pd.Series, path: str, df_row: pd.DataFrame) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"No read permission for {path}")
    df_row["Hogwarts House"] = predicted_houses
    df_row.to_csv(path, 
                  mode="w", 
                  index=False, 
                  header= True)

def calculate_probabilities(theta: pd.DataFrame , data: pd.DataFrame) -> npt.NDArray[np.float64]:
    z = data.to_numpy() @ theta.to_numpy().T
    p = lrt.sigm(z)
    return p

def loop_for_all_house_and_predict_probabilities(theta: pd.DataFrame , data: pd.DataFrame) -> pd.DataFrame:
    all_probabilities = pd.DataFrame(index=data.index)
    houses = theta.loc[:, "House"].unique().tolist()
    data = data.dropna(axis=1, how='any')
    for house in houses:
        p = calculate_probabilities(theta.loc[theta.loc[:, 'House'] == house, :].drop(columns=['House']), data.loc[:, :])
        all_probabilities[house] = p
    predicted_houses = all_probabilities.idxmax(axis=1)
    print(predicted_houses)
    return predicted_houses

def main(path_to_test_file: str, path_to_weight_file: str) -> int:
    try:
        data_test: pd.DataFrame = lrt.charge_file(path_to_test_file)
        theta: pd.DataFrame = lrt.charge_file(path_to_weight_file)
        data_cleaning_test, houses_test, materie_test = lrt.cleaning(data_test)
        houses_test  = np.full(data_cleaning_test.shape[0], np.nan)
        data_std_test = lrt.std_all_input_value(data_cleaning_test)
        predicted_houses = loop_for_all_house_and_predict_probabilities( theta, data_std_test)
        write_int_test_file(predicted_houses, path_to_test_file, data_test)
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
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <path_to_test_file> <path_to_weight_file>")
        sys.exit(1)
    path_to_test_file = sys.argv[1]
    path_to_weight_file = sys.argv[2]
    sys.exit(main(path_to_test_file, path_to_weight_file))