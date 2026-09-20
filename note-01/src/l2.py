import numpy as np
import pandas as pd
import time

SEED = 20260920


def withIntercept(features: np.ndarray) -> np.ndarray:
    """Prepend the x0 = 1 column: (m,) or (m, n) -> (m, n + 1)."""
    if features.ndim == 1:
        features = features[:, None]
    return np.column_stack([np.ones((features.shape[0], 1)), features])

# AI Generated
def makeRandom(
        m: int,
        n: int,
        seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random linear data y = X @ theta + N(0, 1). Returns (features, y, theta)."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((m, n))
    theta = rng.standard_normal(n + 1)
    y = withIntercept(features) @ theta + rng.standard_normal(m)
    return features, y, theta


df = pd.read_csv('../data/lin1d.csv')
x = df['x'].to_numpy()
y = df['y'].to_numpy()
X = withIntercept(x)
 
t1 = time.time()
print(np.linalg.inv(X.T @ X) @ X.T @ y)
print(f'took {time.time() - t1: .3f} to calculate')
print()

t1 = time.time()
print(np.linalg.solve(X.T @ X, X.T @ y))
print(f'took {time.time() - t1: .3f} to calculate')
print()

t1 = time.time()
print(np.linalg.lstsq(X, y, rcond=None))
print(f'took {time.time() - t1: .3f} to calculate')
print()

# AI Generated 
# L2.3 random data: m = 100_000 examples, n = 10 and n = 2000 features.
# Kept in memory, not on disk: the n = 2000 case is ~1.5 GiB as float64 and
# would be ~3.7 GiB written out as CSV. Same SEED => same data every run.
smallFeatures, smallY, smallTheta = makeRandom(100_000, 10)
bigFeatures, bigY, bigTheta = makeRandom(100_000, 2000)

# print(f"random n=10:   features {smallFeatures.shape} -> X {smallFeatures.shape[0], smallFeatures.shape[1] + 1}")
# print(f"random n=2000: features {bigFeatures.shape} -> X {bigFeatures.shape[0], bigFeatures.shape[1] + 1}")

def test(features, y, theta):
    X = withIntercept(features)

    print("inv method")
    t1 = time.time()
    theta = (np.linalg.inv(X.T @ X) @ X.T @ y)
    print(f'took {time.time() - t1: .3f} to calculate')
    print()

    print('solve method')
    t1 = time.time()
    theta = (np.linalg.solve(X.T @ X, X.T @ y))
    print(f'took {time.time() - t1: .3f} to calculate')
    print()

    print('lstsq method')
    t1 = time.time()
    theta = (np.linalg.lstsq(X, y, rcond=None))
    print(f'took {time.time() - t1: .3f} to calculate')
    print()

print('starting test on n = 10')
test(smallFeatures, smallY, smallTheta)
print('starting test on n = 2000')
test(bigFeatures, bigY, bigTheta)