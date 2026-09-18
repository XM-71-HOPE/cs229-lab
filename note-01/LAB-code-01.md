# Lab 01 — CS229 Notes 1, implemented

**Covers.** Everything in notes 1, §1–§4: the cost function, batch and stochastic
gradient descent, the normal equations, feature scaling, and locally weighted
regression. Plus the logistic regression on §5, because it is the natural next
step and its failure mode is more interesting than its success.

**Goal.** You have read these algorithms. This lab makes you *build* them, and
then makes each one check another, so you never need an answer key to know
whether you got it right. Where two methods must agree, the agreement is your
test.

**Setup.** Python 3, `numpy`, `matplotlib`. Put the CSVs from `data/` next to
your code, or point at them with a path. Everything is small — no GPU, no
downloads, no internet.

**Rules.** No solutions included. Hand back the repo (code + a short
`answers.md` with your numbers and one-line explanations) and I grade it.

**Difficulty.** ★ warm-up · ★★ the real work · ★★★ where it bites.

**Time.** Roughly 4–6 hours total. Each problem is labelled. You may hand back a
partial run at any point; a missing problem costs nothing.

---

## L1 ★ — batch gradient descent, one feature

Data: `data/lin1d.csv` (columns `x`, `y`).

1. Implement the cost

   ```
   J(θ) = (1/2) Σᵢ (θ₀ + θ₁xᵢ − yᵢ)²
   ```

   and its gradient. Derive the gradient by hand first, then code it.
2. Implement batch gradient descent. All components of `θ` update in the same
   step, using the *same* old `θ`. Initialise `θ = 0`.
3. Report the `θ` you converge to, to 6 decimals, with `α = 0.01`.
4. Sweep `α` over a log grid from `1e-4` to `10`. For each, record whether `J`
   decreases on every single step.
   Report the largest `α` that stays monotone, and the smallest that diverges.
5. Plot `J` vs iteration for three values: one safely small, one near the
   boundary, one past it.

**What you are testing.** The notes say convergence holds "assuming the learning
rate α is not too large" and move on. You are finding that boundary yourself.

---

## L2 ★★ — the closed form, and where it stops being usable

Same data as L1, plus two random datasets you generate yourself.

1. Derive `θ = (XᵀX)⁻¹Xᵀy` from `∇_θ J = 0`. Then implement it **three ways**:
   - `np.linalg.inv(X.T @ X) @ X.T @ y`
   - `np.linalg.solve(X.T @ X, X.T @ y)`
   - `np.linalg.lstsq(X, y, rcond=None)`

   Remember the `x₀ = 1` column. It is easy to forget and it silently costs you
   the intercept.
2. Confirm all three agree with L1's converged `θ` to at least 6 decimals. This
   is your correctness check for both problems at once.
3. Generate random data with `m = 100_000`, `n = 10`. Time all three. Then
   repeat with `n = 2000` (features, not examples). Report wall-clock time for
   each, and the growth in time as `n` goes from 10 to 2000.
4. One line: at roughly what `n` would you stop using the closed form? Why does
   L1 exist at all?

**What you are testing.** The `O(n³)` in `(XᵀX)⁻¹` is a fact you have read. This
is the fact becoming a stopwatch reading.

---

## L3 ★★ — stochastic gradient descent, and the noise it leaves behind

Same data.

1. Implement the stochastic update: one training example at a time, parameters
   updated after each. Loop over the data in a random order, several times.
2. Plot `J(θ)` measured after every single example, over the first few epochs.
   It should not be monotone. Say in one line why that is expected and not a bug.
3. Compare the final `θ` with L2's answer. Report the gap.
4. Run with 5 different shuffles. Report the spread of the final `θ` across runs.
   Which component wanders more, and why?
5. Implement the decaying learning rate the notes mention in footnote 2. Report
   whether the spread across the 5 shuffles shrinks. One line on why.

---

## L4 ★★ — feature scaling, measured

Data: `data/lin2d.csv` (columns `size`, `bedrooms`, `price`).

1. Run batch GD on the raw features. Report iterations until `|ΔJ| < 1e-6`, and
   find the largest stable `α`.
2. Standardise each feature (subtract mean, divide by standard deviation).
   Repeat. Report iterations and the largest stable `α`.
3. Report the ratio of iteration counts. Then report the ratio of stable `α`.
4. Look back at the contour plot on page 6 of the notes. One line connecting
   that picture to your two numbers: why does stretching one axis break
   gradient descent?

Do **not** standardise before fitting if you want the raw-fit numbers to mean
anything. Keep the two runs separate.

---

## L5 ★★ — logistic regression, and its failure mode

Data: `data/class1d.csv` (columns `x`, `y`, with `y ∈ {0, 1}`).

Note the shape of the data before you fit it: most points lie in `[-3, 3]`, and
four points sit far out on the right. That is deliberate.

1. Fit three models:
   - (a) **linear** regression on the 0/1 labels
   - (b) **logistic** regression by gradient *ascent* on the log-likelihood
   - (c) **logistic** regression by Newton's method
2. For each, report the decision boundary, i.e. the `x` where `h(x) = 0.5`.
   Plot all three `h(x)` on one axis, with the data.
3. The four far-out points carry a lot of leverage. Say what they do to the
   linear fit's boundary, and why they do not do the same to the logistic one.
   This is the whole reason people stopped using least squares for classification.
4. The data is linearly separable. Report `‖θ‖` after 10, 100 and 10 000
   iterations of gradient ascent. Something is wrong, and it is not your code.
   Explain it in two sentences.
5. Add a penalty `λ‖θ‖²` to the objective. Report `‖θ‖` for
   `λ ∈ {0.01, 0.1, 1, 10}`. One line: what did `λ` actually buy you?
6. Newton's method vs gradient ascent: report iterations to reach the same
   tolerance. Report the ratio. One line on where the extra cost per iteration
   went.

---

## L6 ★★★ — locally weighted regression, and your own cache idea, measured

Data: `data/nonlin1d.csv` (columns `x`, `y`). The curve is
`sin(1.5x) + 0.25x` plus a little noise.

1. Fit plain linear regression. Plot it against the data, and say in one line
   whether it underfits or overfits.
2. Implement LWR. For each query point, build the diagonal weight matrix `W`,
   solve `θ = (XᵀWX)⁻¹XᵀWy`, output `θᵀx`. Fit at 200 evenly spaced query
   points, for `τ ∈ {1.0, 0.3, 0.1, 0.03}`. Plot all four curves with the data,
   and label which `τ` underfits and which overfits.
3. Now the cache. At `x₀ = 0.5`, fit once and keep `θ`. For `δ` from `0.001` to
   `1.5`, predict at `x₀ + δ` using the **cached** `θ`, and compare against a
   **fresh** fit at `x₀ + δ`. Plot `|error|` vs `δ`, one line per `τ`.
4. For each `τ`, report the largest `δ` at which the cached prediction is within
   1% of a fresh one. Put the four numbers in a table.
5. The interesting part: pick a `δ` in the middle and show that the cache is
   cheap in one region of the curve and expensive in another *at the same `δ`*.
   Report both errors and explain the difference.
6. One line each:
   - how many cached fits would cover `[-4, 4]` at your 1% tolerance for `τ = 0.1`?
   - what happens to that count with 20 features instead of 1?

**What you are testing.** You proposed caching a fit rather than refitting. This
problem measures what that costs, which is the only way to know whether the idea
pays.

---

## L7 ★ — spot the bug

This is meant to be batch gradient descent.

```python
def batch_gd(X, y, theta, alpha, iters, m):
    for _ in range(iters):
        for j in range(len(theta)):
            pred = X @ theta
            err  = pred - y
            theta[j] = theta[j] - alpha * (X[:, j] @ err) / m
    return theta
```

1. Say what it actually computes, and name the thing it accidentally became.
2. There is one line in the notes that warns about exactly this. Quote the idea
   (not the exact words) and point at the line of code that violates it.
3. Construct a dataset where the difference between this and correct batch GD is
   visible, and report how the two trajectories differ. Make the difference
   large on purpose.

---

## Handing back

Repo layout I'd like to see:

```
cs229-notes1-lab/
  data/            (as given)
  src/             your code
  answers.md       your numbers + the one-line explanations
```

`answers.md` is where the real signal is. Numbers without a sentence next to
them are only half the answer.

Send it over when it runs, or when you are stuck on one and want to move on. I
will read the code, not just the outputs.
