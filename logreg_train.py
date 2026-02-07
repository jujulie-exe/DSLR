from typing import Mapping
import numpy as np
import pandas as pd
import os
from typing import Final
import numpy.typing as npt
import random
import matplotlib as plt
import matplotlib.pyplot as plt


PATH_DATA_SET: Final = "datasets/dataset_train.csv"
ITERATION_RANGE : Final = 1000
LEARNING_RATE: Final = 0.1
LIMIT_STEP_RATE: Final = 0.01
PATH_OUTPUT_FILE: Final = "weight.csv"

def matrix_learning_graf(step, history: dict) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    for ax, (house, data) in zip(axes.flat, history.items()):
        y = np.array(data["loss cost"], dtype=float)
        #y = data["loss cost"]
        x = arr = np.arange(len(y))
        ax.scatter(x, y)
        ax.set_title(f"{house} – p")
    plt.tight_layout()
    plt.show()

def write_in_file(theta: npt.NDArray[np.float64], house: str, ind: int, path: str, materie: list[str]):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"No read permission for {path}")
    if ind == 0:
        flag = 'w'
    else:
        flag = 'a'
    data_dict = {'House': house}
    for i, nome in enumerate(materie):
        data_dict[nome] = theta[i]
    df_row = pd.DataFrame([data_dict])
    df_row.to_csv(path, 
                  mode=flag, 
                  index=False, 
                  header= not ind)

def stadarrizzazione(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mean = x.sum() / len(x)
    new_array : npt.NDArray[np.float64]
    sigma = stdDeV(x)
    new_array = (x - mean) / sigma
    return new_array

def stdDeV(x: npt.NDArray[np.float64]):
    n = len(x)
    mean = x.sum() / n
    varianza = np.sum((x - mean)**2) / n
    return np.sqrt(varianza)

def denormalizzare(theta_std: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    sigma = stdDeV(x)
    #mu = x.sum() / len(x)
    theta_deno = theta_std / sigma
    #intercetta = intercetta_std - (pendenza_std * mu / sigma)
    return theta_deno

def charge_file( path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"No read permission for {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty")
    return df

def sigm(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p: npt.NDArray[np.float64] = 1 / (1 + np.exp(-z))
    return p

def derivate_loss_fun(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64], data :pd.DataFrame) -> npt.NDArray[np.float64]:
    m = data.shape[0]
    error = (p - y)
    d = ((1/m) * (error @ data.to_numpy()))
    return d

#def derivate_loss_for_dependet_theta(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64], data: pd.DataFrame) -> npt.NDArray[np.float64]:
#    m = data.shape[0]
#    d = ((1/m) * np.sum(p - y))
#    return d

def calculate_step(d: npt.NDArray[np.float64], rate: float ) -> npt.NDArray[np.float64]:
    return rate * d

def cheack_value_step(s: npt.NDArray[np.float64]) -> np.bool_:
    #return np.linalg.norm(s) <= LIMIT_STEP_RATE
    return np.all(s <= LIMIT_STEP_RATE)

def trace_history_loss_cost(history : dict, loss_cost: np.float64):
    history["loss cost"].append(loss_cost)
    #mi sa che mi serve solo il loss cost e salvare l'ultimo theta

def trace_history_theta(history : dict, theta: npt.NDArray[np.float64]):
    history["theta"].append(theta)

def trace_history_p(history : dict, p: npt.NDArray[np.float64]):
    history["p"].append(p)

def loss_logistic(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    m = len(y)
    epsilon = 1e-15
    loss = -(1/m) * np.sum(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))
    return loss


def gradient_descend(data: pd.DataFrame, y: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], dict]:
    theta: npt.NDArray[np.float64]= np.zeros(data.shape[1])
    history = {"theta": [], "loss cost": [], "p": [], "step": []}
    for _ in range(ITERATION_RANGE):
        z = data.to_numpy() @ theta
        p = sigm(z)
        loss_cost = np.sum(p - y)
        loss_cost = loss_logistic(p, y)
        derivate = derivate_loss_fun(p, y, data) 
        step = calculate_step(derivate, LEARNING_RATE)
        trace_history_loss_cost(history, loss_cost)
        if (cheack_value_step(step)):
            break
        theta = theta - step
    trace_history_theta(history, theta)
    trace_history_p(history, p)
    return theta, history

def cleaning(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series , list[str]]:
    new_data = data.select_dtypes(include=[np.number]).copy()
    materie = data.columns[6:].tolist()
    materie.append("Bias")
    house = pd.Series()
    #TODO verificare da che collona prendere i dati
    if 'Index' in new_data.columns:
        new_data = new_data.drop(columns=['Index'])
    if 'Birthday' in new_data.columns:
        new_data = new_data.drop(columns=['Birthday'])
    if 'Best Hand' in new_data.columns:
        new_data = new_data.drop(columns=['Best Hand'])
    if 'Hogwarts House' in data.columns:
        house = data['Hogwarts House'].copy()
    if new_data.empty or house.empty:
        raise ValueError("Dataset is empty")
    return new_data, house, materie


def std_all_input_value(data: pd.DataFrame) -> pd.DataFrame:
    new_data = pd.DataFrame()
    for i in range(0, data.shape[1]):
        data.iloc[:, i] = data.iloc[:, i].fillna(data.iloc[:, i].mean())
        std = stadarrizzazione(data.iloc[:, i].to_numpy())
        new_data[data.columns[i]] = std;
    new_data["Bias"] = np.full(data.shape[0], 1)
    return new_data

def for_loop_gradiend_descent(houses: pd.Series , data_std: pd.DataFrame, materie: list[str]) -> None:
    unique_house = houses.unique().tolist()
    all_probabilities = pd.DataFrame(index=data_std.index)
    history: dict[str, dict[str, float]] = {}

    for index, house in enumerate(unique_house): 
        weight: npt.NDArray[np.float64] = np.where(houses == house, 1, 0)
        theta, historyTmp = gradient_descend(data_std, weight)
        final_p_array = historyTmp['p'][-1]
        all_probabilities[house] = final_p_array
        
        history[house] = historyTmp
        write_in_file(np.array(theta), house, index, PATH_OUTPUT_FILE, materie)

    predicted_houses = all_probabilities.idxmax(axis=1)
    correct = (predicted_houses == houses).sum()
    total = len(houses)
    print(f"Accuracy: {correct / total:.2%} ({correct}/{total})")
    errors = houses[predicted_houses != houses]
    #print(errors)
    matrix_learning_graf(1000, history);

def main() -> None:
    data_raw: pd.DataFrame = charge_file(PATH_DATA_SET)
    data_cleaning, houses, materie = cleaning(data_raw)
    data_std = std_all_input_value(data_cleaning)
    for_loop_gradiend_descent(houses, data_std, materie)
    return None

if __name__ == "__main__":
    main()
