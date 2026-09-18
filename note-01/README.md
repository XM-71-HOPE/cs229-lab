# cs229-notes1-lab

Two pieces of work built from CS229 Notes 1 (§1–§5).

| File | What it is |
| --- | --- |
| `PS-math-01.md` | Problem set on the math. Paper. No computer needed. |
| `LAB-code-01.md` | Lab spec. You write the code. |
| `data/` | The four datasets the lab uses, plus the generator that made them. |
| `data/make_data.py` | `python3 data/make_data.py` regenerates every CSV. Stdlib only. |

## Datasets

| File | Rows | Columns | Built from |
| --- | --- | --- | --- |
| `lin1d.csv` | 60 | `x, y` | `y = 3 + 1.5x`, noise σ = 2 |
| `lin2d.csv` | 120 | `size, bedrooms, price` | `price = 80 + 0.14·size + 12·bedrooms`, noise σ = 25 |
| `class1d.csv` | 144 | `x, y` | `y = 1` iff `x > 0`; bulk in [−3, 3], four far points on the right |
| `nonlin1d.csv` | 240 | `x, y` | `y = sin(1.5x) + 0.25x`, noise σ = 0.12 |

Seed 20260916. Everything is small enough to plot by eye.

## Reading order

Do `PS-math-01.md` first, or at least Part B of it. The lab assumes you have
already worked through why least squares is a consequence of Gaussian noise
rather than a law of nature. That is the difference between coding the cost
function and knowing what you coded.

## Deliverable

```
src/         your code
answers.md   your numbers, each with a one-line explanation
```

Hand back when it runs, or when you are stuck on one and want to move on.
Partial is fine.
