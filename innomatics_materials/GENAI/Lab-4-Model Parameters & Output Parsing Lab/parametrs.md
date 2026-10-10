# Temperature in Large Language Models (LLMs)

The **temperature** parameter controls **how random the next token selection is** by modifying the probability distribution before sampling.

## Step 1: Raw Model Output (Logits)

The LLM first outputs **logits** (unnormalized scores), not probabilities.

Example:

| Token | Logit |
|--------|------:|
| Paris | 8 |
| London | 4 |
| Berlin | 3 |
| Apple | 1 |

---

## Step 2: Softmax Without Temperature

The logits are converted into probabilities using the Softmax function:

\[
P_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

where:

- \(z_i\) = logit of token \(i\)
- \(P_i\) = probability of token \(i\)

For the example:

\[
z=[8,4,3,1]
\]

Exponentials:

\[
e^8=2980.96
\]

\[
e^4=54.60
\]

\[
e^3=20.08
\]

\[
e^1=2.72
\]

Sum:

\[
3058.36
\]

Probabilities:

| Token | Probability |
|--------|------------:|
| Paris | 0.974 |
| London | 0.018 |
| Berlin | 0.007 |
| Apple | 0.001 |

---

## Step 3: Introducing Temperature

Temperature modifies the logits before applying Softmax.

The new equation becomes:

\[
P_i=
\frac{
e^{z_i/T}
}
{
\sum_j e^{z_j/T}
}
\]

where:

- \(T\) = temperature

---

## Case 1: Temperature = 1

Nothing changes.

\[
z/T=z
\]

The probabilities remain the same.

---

## Case 2: Temperature = 0.5

Divide each logit by 0.5:

\[
[8,4,3,1]
\rightarrow
[16,8,6,2]
\]

Exponentials:

| Token | \(e^{z/T}\) |
|--------|------------:|
| Paris | 8,886,111 |
| London | 2,981 |
| Berlin | 403 |
| Apple | 7.39 |

Resulting probabilities:

| Token | Probability |
|--------|------------:|
| Paris | 99.96% |
| London | 0.03% |
| Berlin | 0.004% |
| Apple | ~0% |

The distribution becomes much sharper.

---

## Case 3: Temperature = 2

Divide logits by 2:

\[
[8,4,3,1]
\rightarrow
[4,2,1.5,0.5]
\]

Exponentials:

| Token | Value |
|--------|------:|
| Paris | 54.60 |
| London | 7.39 |
| Berlin | 4.48 |
| Apple | 1.65 |

Sum:

\[
68.12
\]

Probabilities:

| Token | Probability |
|--------|------------:|
| Paris | 80% |
| London | 11% |
| Berlin | 7% |
| Apple | 2% |

The distribution becomes flatter, making sampling more diverse.

---

## Visual Interpretation

### Temperature = 0.2

```text
Paris    ███████████████████████

London   ▏

Berlin   ▏

Apple    ▏
```

Very confident.

### Temperature = 1

```text
Paris    ███████████████

London   ██

Berlin   █

Apple    ▏
```

Normal distribution.

### Temperature = 2

```text
Paris    ████████

London   ███

Berlin   ██

Apple    █
```

Much flatter distribution.

---

## Why Dividing by Temperature Works

Original logits:

```text
8
4
```

Difference:

```text
4
```

With Temperature = 0.5:

```text
16
8
```

Difference:

```text
8
```

The gap becomes larger, making Softmax more confident.

With Temperature = 2:

```text
4
2
```

Difference:

```text
2
```

The gap becomes smaller, producing a flatter probability distribution.

---

## Extreme Cases

### Temperature → 0

\[
\lim_{T\to0}
\text{Softmax}(z/T)
\]

Only the largest logit survives.

```text
Paris = 1
Everything else = 0
```

This is effectively **greedy decoding**.

---

### Very Large Temperature

As \(T\) becomes very large, all logits shrink toward zero.

The probability distribution approaches a uniform distribution.

```text
Paris    27%

London   25%

Berlin   24%

Apple    24%
```

The output becomes almost random.

---

## Entropy Perspective

Temperature directly controls the entropy of the probability distribution.

| Temperature | Entropy | Behavior |
|-------------|---------|----------|
| Low | Low | Deterministic |
| Medium | Medium | Balanced |
| High | High | Creative / Random |

---

## Summary

| Temperature | Effect on Logits | Distribution | Behavior |
|-------------|------------------|-------------|----------|
| 0 (special case) | Greedy selection | Single token | Deterministic |
| 0.2 | Amplifies logits | Very sharp | Highly confident |
| 0.5 | Larger gaps | Sharp | Less random |
| 1.0 | Original logits | Original | Default |
| 1.5 | Smaller gaps | Flatter | More diverse |
| 2.0 | Much smaller gaps | Even flatter | More creative |
| Very large | Logits approach zero | Nearly uniform | Almost random |

---

# Key Formula

\[
\boxed{
P_i =
\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
}
\]

---

# Top-p (Nucleus Sampling)

Top-p sampling does **not modify probabilities**.

Instead, it chooses the smallest set of tokens whose cumulative probability exceeds a threshold \(p\), then samples only from that subset.

---

## Step 1: Original Probability Distribution

| Token | Probability |
|--------|------------:|
| Paris | 0.50 |
| London | 0.20 |
| Berlin | 0.15 |
| Rome | 0.08 |
| Madrid | 0.04 |
| Apple | 0.03 |

---

## Step 2: Sort Tokens

| Rank | Token | Probability | Cumulative |
|------|--------|------------:|-----------:|
| 1 | Paris | 0.50 | 0.50 |
| 2 | London | 0.20 | 0.70 |
| 3 | Berlin | 0.15 | 0.85 |
| 4 | Rome | 0.08 | 0.93 |
| 5 | Madrid | 0.04 | 0.97 |
| 6 | Apple | 0.03 | 1.00 |

---

## Step 3: Choose Top-p

Suppose:

\[
p=0.90
\]

We keep adding tokens until cumulative probability reaches or exceeds 0.90.

Selected tokens:

- Paris
- London
- Berlin
- Rome

The remaining tokens are discarded.

---

## Step 4: Renormalize

The selected probabilities sum to 0.93.

We renormalize them:

\[
P_i'=\frac{P_i}{\sum_{j\in S}P_j}
\]

where \(S\) is the selected set.

Result:

| Token | Old Probability | New Probability |
|--------|----------------:|----------------:|
| Paris | 0.50 | 0.538 |
| London | 0.20 | 0.215 |
| Berlin | 0.15 | 0.161 |
| Rome | 0.08 | 0.086 |

---

## Mathematical Formula

Given:

\[
P=(p_1,p_2,\ldots,p_n)
\]

Find the smallest set \(S\) such that:

\[
\sum_{i\in S}p_i \ge p
\]

Then:

\[
P_i'=
\begin{cases}
\dfrac{P_i}{\sum_{j\in S}P_j}, & i\in S\\[8pt]
0, & i\notin S
\end{cases}
\]

---

## Example 1: Top-p = 0.5

Only Paris is kept.

```text
Paris = 100%
```

---

## Example 2: Top-p = 0.8

Selected tokens:

- Paris
- London
- Berlin

Renormalized probabilities:

| Token | New Probability |
|--------|----------------:|
| Paris | 0.588 |
| London | 0.235 |
| Berlin | 0.177 |

---

## Example 3: Top-p = 0.99

Almost every token remains.

Behavior becomes similar to sampling from the full distribution.

---

## Visual Interpretation

Original distribution:

```text
Paris      ██████████ 50%

London     ████       20%

Berlin     ███        15%

Rome       ██          8%

Madrid     █           4%

Apple      █           3%
```

Top-p = 0.8:

```text
✓ Paris

✓ London

✓ Berlin

✗ Rome

✗ Madrid

✗ Apple
```

Only the selected tokens are sampled from.

---

## Why It Is Called Nucleus Sampling

The highest-probability tokens form the "nucleus" of the probability distribution.

```text
Entire Vocabulary

[ Paris London Berlin Rome ]   <-- Nucleus

Madrid Apple Tiger Moon ...
```

Only this nucleus participates in sampling.

---

## Top-k vs Top-p

Distribution:

| Token | Probability |
|--------|------------:|
| A | 0.60 |
| B | 0.20 |
| C | 0.10 |
| D | 0.05 |
| E | 0.03 |
| F | 0.02 |

### Top-k = 3

Always selects:

- A
- B
- C

### Top-p = 0.9

Also selects:

- A
- B
- C

Now consider another distribution:

| Token | Probability |
|--------|------------:|
| A | 0.90 |
| B | 0.03 |
| C | 0.02 |
| D | 0.02 |
| E | 0.02 |
| F | 0.01 |

Top-k = 3 selects:

- A
- B
- C

Top-p = 0.9 selects:

- A

Top-p adapts to the model's confidence.

---

## Temperature vs Top-p

| Feature | Temperature | Top-p |
|----------|-------------|--------|
| Operates on | Logits | Probabilities |
| Changes probabilities | ✅ Yes | ❌ No (only renormalizes) |
| Changes candidate set | ❌ No | ✅ Yes |
| Controls | Randomness | Diversity |
| Mathematical operation | Divide logits by \(T\) | Filter by cumulative probability |

---

## Complete Decoding Pipeline

Most LLMs perform decoding in the following order:

1. Model outputs logits.
2. Apply Temperature:

\[
z_i'=\frac{z_i}{T}
\]

3. Apply Softmax:

\[
P_i=\frac{e^{z_i'}}{\sum_j e^{z_j'}}
\]

4. Apply Top-p filtering.
5. Renormalize probabilities.
6. Sample one token.

---

## Final Comparison

| Parameter | Purpose | Mathematical Effect | Result |
|-----------|---------|---------------------|--------|
| Temperature | Controls randomness | Rescales logits before Softmax | Changes probability distribution |
| Top-p | Controls diversity | Keeps only tokens whose cumulative probability exceeds \(p\) | Changes candidate token set |
| Top-k | Fixed candidate count | Keeps exactly \(k\) highest-probability tokens | Fixed-size candidate set |