import numpy as np
import pandas as pd
import copy

df = pd.read_csv("../data/lin1d.csv")

class Hypothesis:
    theta0: float
    theta1: float

    def calculate(self, x: float) -> float:
        return self.theta0 + self.theta1 * x

    def __init__(self, theta0 : float, theta1 : float):
        self.theta1 = theta1
        self.theta0 = theta0
        return

def computeJ(h: Hypothesis, df:pd.DataFrame) -> float:
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    res = 0.
    for xi, yi in zip(x, y):
        res += 1/2 * (h.calculate(xi) - yi) ** 2
    return res


def batchUpdate(
        h: Hypothesis,
        alpha: float,
        data : pd.DataFrame,
        tolerance: float,
) -> bool:
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    dataSize = data.shape[0]
    _h = copy.copy(h)
    for xi, yi in zip(x, y):
        h.theta0 -= alpha / dataSize * (_h.calculate(xi) - yi)
        h.theta1 -= alpha / dataSize * (_h.calculate(xi) - yi) * xi
    if (
        abs(_h.theta1 - h.theta1) <= tolerance 
        and abs (_h.theta0 - h.theta0) <= tolerance
    ):
        return True
    return False

def runBatchGD(
        h: Hypothesis,
        alpha : float, 
        maxIterations : int,
        tolerance : float,
        data : pd.DataFrame
    ) -> tuple[Hypothesis, bool, list[float]]:
    t = 0
    costPath = []
    costPath.append(computeJ(h, data))
    while(not batchUpdate(
        h, alpha, data, tolerance
    )):
        t+=1
        costPath.append(computeJ(h, data))
        if t>=maxIterations:
            return h, False, costPath
    return h, True, costPath

    
h, converged, path = runBatchGD(
    Hypothesis(0., 0.),
    1e-3,
    int(2e4),
    1e-6,
    df
)

print("Converged" if converged else "Not converged")
print(f"iterations: {len(path) - 1}")
print(f"theta0: {h.theta0:.6f}")
print(f"theta1: {h.theta1:.6f}")
print(f"J initial: {path[0]:.6f}")
print(f"J final:   {path[-1]:.6f}")
