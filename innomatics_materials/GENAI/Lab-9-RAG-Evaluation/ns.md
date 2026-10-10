# Noise Sensitivity in Ragas

## Definition

**Noise Sensitivity** measures how much a RAG system's generated answer is influenced by **irrelevant (noisy) retrieved documents**.

A good RAG system should ignore irrelevant context and generate answers using only the relevant retrieved information.

---

# Mathematical Formulation

Let:

- \(Q\) = User Question
- \(C = \{c_1, c_2, \ldots, c_n\}\) = Retrieved contexts
- \(R \subseteq C\) = Relevant contexts
- \(N = C \setminus R\) = Irrelevant (Noise) contexts

Suppose the generated answer contains \(m\) atomic claims:

\[
A = \{a_1, a_2, \ldots, a_m\}
\]

where each \(a_i\) is an individual factual statement.

---

## Step 1: Break the Answer into Claims

Example answer:

> Python was created by Guido van Rossum in 1991 and is one of the largest reptiles.

Claims:

\[
A =
\{
a_1=\text{"Python was created by Guido van Rossum"},
a_2=\text{"Python was created in 1991"},
a_3=\text{"Python is one of the largest reptiles"}
\}
\]

---

## Step 2: Verify Each Claim

For every claim, determine whether it is supported by:

- Relevant context
- Noise context
- Both
- Neither

Example:

| Claim | Relevant Context | Noise Context |
|--------|-----------------|---------------|
| Guido created Python | ✅ | ❌ |
| Created in 1991 | ✅ | ❌ |
| Largest reptiles | ❌ | ✅ |

---

## Step 3: Compute the Score

Let

- \(n_{noise}\) = Number of claims supported **only by noisy contexts**
- \(n_{total}\) = Total number of claims

The conceptual Noise Sensitivity score is

\[
\boxed{
\text{Noise Sensitivity}
=
\frac{n_{noise}}{n_{total}}
}
\]

or equivalently,

\[
\boxed{
\text{Noise Sensitivity}
=
\frac{
\sum_{i=1}^{m}
\mathbf{1}
\left(
a_i \text{ is supported only by noisy context}
\right)
}
{m}
}
\]

where

\[
\mathbf{1}(x)=
\begin{cases}
1,& \text{if }x\text{ is true}\\
0,& \text{otherwise}
\end{cases}
\]

---

# Interpretation

| Score | Interpretation |
|--------|----------------|
| **0.0** | Completely ignores irrelevant information (Best) |
| **0.2** | Slightly affected by noise |
| **0.5** | Half of the answer depends on noisy context |
| **1.0** | Entire answer is generated from irrelevant context (Worst) |

---

# Example 1: Low Noise Sensitivity

### Question

> Who invented Python?

### Retrieved Context

```text
Context 1:
Python programming language was created by Guido van Rossum in 1991.

Context 2:
The Eiffel Tower is located in Paris.

Context 3:
Basketball was invented by James Naismith.
```

### Generated Answer

```text
Python was created by Guido van Rossum in 1991.
```

### Claim Analysis

| Claim | Relevant | Noise |
|--------|----------|-------|
| Guido created Python | ✅ | ❌ |
| Created in 1991 | ✅ | ❌ |

\[
n_{noise}=0
\]

\[
n_{total}=2
\]

\[
\boxed{
\text{Noise Sensitivity}
=
0
}
\]

The model completely ignored the irrelevant documents.

---

# Example 2: Moderate Noise Sensitivity

### Question

> Who invented Python?

### Retrieved Context

```text
Context 1:
Python programming language was created by Guido van Rossum in 1991.

Context 2:
Python snakes are among the largest reptiles.

Context 3:
Basketball was invented by James Naismith.
```

### Generated Answer

```text
Python was created by Guido van Rossum in 1991 and is one of the largest reptiles.
```

### Claim Analysis

| Claim | Relevant | Noise |
|--------|----------|-------|
| Guido created Python | ✅ | ❌ |
| Created in 1991 | ✅ | ❌ |
| Largest reptiles | ❌ | ✅ |

Therefore,

\[
n_{noise}=1
\]

\[
n_{total}=3
\]

\[
\boxed{
\text{Noise Sensitivity}
=
\frac13
=
0.33
}
\]

Approximately **33%** of the generated answer was influenced by irrelevant context.

---

# Example 3: High Noise Sensitivity

### Generated Answer

```text
Python is one of the largest snakes found in Africa.
```

### Claim Analysis

| Claim | Relevant | Noise |
|--------|----------|-------|
| Largest snake | ❌ | ✅ |
| Found in Africa | ❌ | ✅ |

\[
n_{noise}=2
\]

\[
n_{total}=2
\]

\[
\boxed{
\text{Noise Sensitivity}
=
1
}
\]

The answer was entirely based on irrelevant retrieved information.

---

# How Ragas Computes Noise Sensitivity

Modern versions of **Ragas** do not simply count claims using the above formula. Instead, they use an LLM-as-a-Judge pipeline:

1. Retrieve relevant and irrelevant documents.
2. Break the generated answer into atomic claims.
3. Determine which claims are supported by relevant contexts.
4. Identify claims that are supported only by noisy contexts.
5. Aggregate these judgments into a final score between **0** and **1**.

The mathematical formulation above is a conceptual representation of this evaluation process.

---

# Difference Between Faithfulness and Noise Sensitivity

| Metric | What it Measures | Failure Example |
|----------|------------------|-----------------|
| **Faithfulness** | Whether every claim is supported by the retrieved context | Answer says "Python was created in 1985" while context says 1991 |
| **Noise Sensitivity** | Whether the model incorrectly uses irrelevant retrieved documents | Answer mixes facts about Python programming language with Python snakes |

A response can be **faithful** yet still have **high noise sensitivity** if it faithfully reproduces information from an irrelevant retrieved document. Noise Sensitivity specifically evaluates whether the model can ignore distracting context.