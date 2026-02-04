from typing import Mapping
import numpy as np
import pandas as pd
import os
from typing import Final
import numpy.typing as npt

PATH_NAME: Final = "data.csv"
ITERATION_RANGE : Final = 1000
LEARNING_RATE: Final = 0.1
def stadarrizzazione(x: npt.NDArray[np.float64]):
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

def denormalizzare(intercetta_std: float, pendenza_std: float, x: np.array):
    sigma = stdDeV(x)
    mu = x.sum() / len(x)
    pendenza = pendenza_std / sigma
    intercetta = intercetta_std - (pendenza_std * mu / sigma)
    return intercetta, pendenza

def charge_file( path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"No read permission for {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty")
    required_cols = {"km", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if df[["km", "price"]].isnull().any().any():
        raise ValueError("Null values found in km or price")
    if not np.issubdtype(df["km"].dtype, np.number):
        raise TypeError("km must be numeric")
    if not np.issubdtype(df["price"].dtype, np.number):
        raise TypeError("price must be numeric")
    return df

def sigm(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    z = 1 / (1 + np.exp(-z))
    return z

def derivate_loss_fun(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64], data :pd.DataFrame) -> npt.NDArray[np.float64]:
    m = data.shape[0]
    error = (p - y)
    d = 1/m * (error @ data.T)
    return d

def derivate_loss_for_dependet_theta(p: npt.NDArray, y: npt.NDArray, data: pd.DataFrame) -> npt.NDArray:
    m = data.shape[0]
    d = 1/m * np.sum((p - y))
    return d


def gradient_descend(data: pd.DataFrame, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    theta = np.zeros(data.shape[1])
    for _ in range(ITERATION_RANGE):
        z = theta @ data
        p = sigm(z)
        derivate = derivate_loss_fun(p, y, data) # manca la derivata quella non moltiplicate per le materie
        step = calculate_step(derivate, LEARNING_RATE)
        if (cheack_value_step(step)):
            break
        theta = theta - step
    return theta

def main() -> None:
    data: pd.DataFrame = charge_file(PATH_NAME)
    data = cleaning(data)
    data = standardizzazione(data)
    return None

if ___name__ == "main":
    main()
