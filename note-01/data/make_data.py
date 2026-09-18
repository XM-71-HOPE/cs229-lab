#!/usr/bin/env python3
"""Deterministic dataset generator for the CS229 notes-1 lab.

Pure standard library — no numpy needed. Same seed => same files, every time.

    python3 make_data.py

Writes four CSVs next to this script.
"""

import csv
import math
import os
import random

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260916


def write(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {os.path.basename(path):16s} {len(rows):4d} rows")


rng = random.Random(SEED)

# ---------------------------------------------------------------- L1, L2, L3
# One feature, clean linear trend. True line: y = 3 + 1.5x, noise sigma = 2.
rows = []
for _ in range(60):
    x = rng.uniform(-5.0, 5.0)
    y = 3.0 + 1.5 * x + rng.gauss(0.0, 2.0)
    rows.append([f"{x:.6f}", f"{y:.6f}"])
write("lin1d.csv", ["x", "y"], rows)

# ---------------------------------------------------------------------- L4
# Two features on wildly different scales. True line:
#   price = 80 + 0.14*size + 12*bedrooms, noise sigma = 25
rows = []
for _ in range(120):
    size = rng.uniform(900.0, 4200.0)
    beds = rng.randint(1, 5)
    price = 80.0 + 0.14 * size + 12.0 * beds + rng.gauss(0.0, 25.0)
    rows.append([f"{size:.2f}", beds, f"{price:.4f}"])
write("lin2d.csv", ["size", "bedrooms", "price"], rows)

# ---------------------------------------------------------------------- L5
# Separable binary labels: y = 1 iff x > 0. The bulk sits in [-3, 3]; a few
# points sit far out on the right. Separability is deliberate: it is what makes
# the logistic MLE run off to infinity.
rows = []
for _ in range(140):
    x = rng.uniform(-3.0, 3.0)
    rows.append([f"{x:.6f}", 1 if x > 0.0 else 0])
for x in (18.0, 22.0, 26.0, 31.0):
    rows.append([f"{x:.6f}", 1])
write("class1d.csv", ["x", "y"], rows)

# ---------------------------------------------------------------------- L6
# Smooth nonlinear curve, same one used in math problem C1.
#   y = sin(1.5x) + 0.25x, noise sigma = 0.12
rows = []
for _ in range(240):
    x = rng.uniform(-4.0, 4.0)
    y = math.sin(1.5 * x) + 0.25 * x + rng.gauss(0.0, 0.12)
    rows.append([f"{x:.6f}", f"{y:.6f}"])
write("nonlin1d.csv", ["x", "y"], rows)

print("\nseed:", SEED)
