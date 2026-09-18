# Problem Set 01 — CS229 Notes 1, §2–§4

**Covers.** Matrix-derivative notation, the least-squares cost function, the
probabilistic interpretation of regression (Gaussian noise → likelihood → log),
and the non-parametric turn in locally weighted regression.

**Rules.** No answers here, on purpose. Work them, hand them back, and I grade
them line by line. Partial work is fine and honest — hand in whatever you have.

**Difficulty.** ★ checking your footing · ★★ a real step · ★★★ where it bites.

**Time.** 100–120 minutes if you don't rush. Stop whenever you like.

Written answers, formulas, or scans of paper all work.

---

# Part A — Notation: which symbol answers which question

## A1 ★

Two engineers measure the same quantity `f`, which depends on two adjustable
knobs, `x` and `y`.

The first reports `df/dx = 4`.
The second reports `∂f/∂x = 4`.
Same function, same point, two different symbols.

1. State precisely what each engineer has claimed about the knobs. One sentence
   each, and make the two sentences differ in their *claim*, not just in their
   notation.
2. Give a concrete `f` and a concrete relation between `x` and `y` where the two
   numbers differ. Compute both.
3. In the notes, gradient descent writes `∂J/∂θⱼ` everywhere. What has to be
   true about the `θ`'s for that symbol to be the correct one? Say it as a
   property of the model, not of the symbols.
4. One sentence: in what situation is writing `∂` instead of `d` a mistake, even
   though every step of the arithmetic is clean?

## A2 ★★ — the transpose in the subscript

Let `A` be 2×2 with entries `a₁₁, a₁₂, a₂₁, a₂₂`, and let

```
f(A) = a₁₁² a₂₂ + 3 a₁₁ a₂₁
```

1. Compute all four entries of `∇_A f`.
   The entry rule is: entry `(i,j)` is `∂f/∂A_ij`.
2. Compute all four entries of `∇_{Aᵀ} f`.
   The entry rule is: entry `(i,j)` is `∂f / ∂(Aᵀ)_ij`.
3. Verify `∇_{Aᵀ} f = (∇_A f)ᵀ` on your numbers.
4. The notes state `∇_{Aᵀ} f(A) = (∇_A f(A))ᵀ` with no comment. In one or two
   sentences: why did "differentiating with respect to `Aᵀ`" never need a second
   variable? Point at the exact place the transpose came from.
5. Exhibit a function `f` for which `∇_{Aᵀ} f = ∇_A f`, i.e. no transpose is
   needed at all. What property of `∇_A f` makes that happen?

---

# Part B — Where the cost function comes from

## B1 ★★★ — take the Gaussian out

The notes justify least squares with `y = θᵀx + ε`, `ε ~ N(0, σ²)`.

Replace that assumption. Suppose instead the noise is **uniform on `[−b, b]`**,
with the same `b` for every training example, and `b` known.

1. Write down the likelihood `L(θ)`.
2. Describe exactly what maximizing it selects for. This is not least squares,
   and the geometry of what it optimizes is the whole point.
3. One sentence: what property of the Gaussian is the reason a *square* shows up
   at all? Answer in terms of the shape of the density, not the formula.

## B2 ★★ — the number that is not a probability

Three points: `x = (1, 2, 3)`, `y = (2, 4, 6)`.
Model: `y = θx`, Gaussian noise, `σ = 1`. So

```
p(yᵢ | xᵢ; θ)  =  the density of N(θxᵢ, 1) evaluated at yᵢ
```

1. Compute the log-likelihood `ℓ(θ)` at `θ = 2` and at `θ = 2.5`.
2. Report `L(2)`, `L(2.5)`, and the ratio `L(2)/L(2.5)`.
3. Now measure `y` in cents: `y → 100y`, and correspondingly `σ → 100` (the
   measuring stick changes, the physical situation does not). Recompute `L` at
   `θ = 200` and `θ = 250`, the images of the two values you used in part 2.
4. Which of these survived the change of units: `L` itself, the ratio
   `L(2)/L(2.5)`, the value of `θ` that maximizes `L`?
5. Explain *why* the survivors survived. One or two sentences, phrased in terms
   of what a density actually is.

## B3 ★★ — mean, median, and which noise model each belongs to

Four readings of the same physical quantity: `0, 0, 0, 10`.
Two candidate answers: the mean (2.5) and the median (0).

1. For each candidate, list the four residuals and compute the sum of squares.
2. Same, for the sum of absolute values.
3. The two criteria disagree about which candidate is better. Report which
   criterion picks which candidate.
4. Which criterion is the maximum-likelihood answer under Gaussian noise?
   Under what noise shape would the *other* criterion be the maximum-likelihood
   answer?
5. One sentence: what does this say about least squares as a default? It is a
   choice, and this problem is here to name its price.

## B4 ★★★ — independence is load-bearing

Four readings arrive from one sensor: `0, 0, 0, 10`.
Model: `h(x) = θ`, a single constant. Gaussian noise, `σ = 1`, known.

1. Find the maximum-likelihood `θ`.
2. The sensor glitches and reports one reading twice. You now have
   `0, 0, 0, 0, 10`. Find the new maximum-likelihood `θ`.
3. The two answers differ. Say what the IID assumption just did, and why
   "I saw it twice" is not "I have twice the evidence."
4. Give one real situation where two identical-looking rows are genuinely
   independent evidence, and one where they are not. Be specific about what
   distinguishes them.

## B5 ★★ — a formula wearing a density's clothes

Locally weighted regression uses the weights

```
w⁽ⁱ⁾ = exp( −(x⁽ⁱ⁾ − x)² / (2τ²) )
```

which looks, character for character, like the shape of a Gaussian density.
The notes add a footnote warning you that it is not one.

Give **two independent reasons** it is not a density, and for each one say what
goes wrong if you forget it. Concrete, not a restatement of the footnote.

---

# Part C — The non-parametric turn

## C1 ★★★ — the price of a cache

You fit LWR at a query point `x₀`, keep `θ`, and then a later query arrives at
`x₀ + δ`. You reuse the old `θ` instead of refitting.

Let `L(x) = θ₀ + θ₁x` be the cached line, and let `h` be the curve LWR would give
you if you refit at every query point. The error below is `L(x₀+δ) − h(x₀+δ)`.

1. Suppose the cached line sits at the right height *and* points the right way:
   `L(x₀) = h(x₀)` and `L'(x₀) = h'(x₀)`.
   Show that the prediction error at `x₀ + δ` is **second order** in `δ`, and
   name the derivative that governs it.
2. Check the argument against a degenerate case: what does the error do at *any*
   `δ` when `h` is a straight line? Say why it had to come out that way. What
   does that tell you about the first derivative's role here?
3. Now break the assumption in part 1. Name the **two** situations in which a
   first-order term appears instead, one originating in the data and one in the
   fitting. For each, write what the error term looks like and which symbol sets
   its size.
4. On this curve —

   ```
   x from −4 to 4,  y = sin(1.5x) + 0.25x
   ```

   — say where the *same* `δ` is nearly free and where it is expensive. Locate
   those places by the shape of the curve, not by the fact that it is a sine.
   Then check your claim against the quantity from part 1.
5. `τ` controls how fast the local model changes between nearby queries. In
   words (no plot needed), how does the safe `δ` scale with `τ`? Then say where
   that scaling breaks down, at both ends.

## C2 ★ — a small one, worth checking

The notes' cost function carries a ½ and drops a constant:

```
J(θ) = ½ Σᵢ (h(x⁽ⁱ⁾) − y⁽ⁱ⁾)²
```

The maximum-likelihood derivation in §3 produces the same minimizer with no ½
anywhere in sight, and quietly discards `m·log(1/(√(2π)σ))` on the way.

1. Show in one line that the ½ cannot move the minimizer.
2. The ½ was not there for that reason, though. Name the reason, and where it
   pays off two pages later.
3. The dropped constant gets discarded for a *different* argument than the ½.
   State both arguments and say how they differ.

---

Hand it back when you're done, or when you're stuck. Line-by-line, and I'll say
which of these you actually own and which only look familiar.
