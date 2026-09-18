import numpy as np
import pandas as pd
import copy

df = pd.read_csv("../data/lin1d.csv")
x = df["x"].to_numpy()
y = df["y"].to_numpy()
convergeEpisilon = 1e-6

class Hypothesis:
    theta0: float
    theta1: float

    def calculate(self, x: float) -> float:
        return self.theta0 + self.theta1 * x

    def __init__(self, theta0, theta1):
        self.theta1 = theta1
        self.theta0 = theta0
        return

h = Hypothesis(0., 0.)
alpha = 2e-5

def batchUpdate() -> bool:
    global h, alpha, convergeEpisilon
    _h = copy.copy(h)
    for xi, yi in zip(x, y):
        h.theta0 -= alpha * (_h.calculate(xi) - yi)
        h.theta1 -= alpha * (_h.calculate(xi) - yi) * xi
    if (
        abs(_h.theta1 - h.theta1) <= convergeEpisilon 
        and abs (_h.theta0 - h.theta0) <= convergeEpisilon
    ):
        return True
    return False

t = 0
T = 10000
while(not batchUpdate()):
    t+=1
    if t>=T:
        print("Not converging after " + str(T) + " iterations!")
        print(f"theta0: {h.theta0:.6f}")
        print(f"theta1: {h.theta1:.6f}")
        exit(1)

print("Converged after " + str(t) + " iterations!")
print(f"theta0: {h.theta0:.6f}")
print(f"theta1: {h.theta1:.6f}")