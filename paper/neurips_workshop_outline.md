# StableSelect: Deployment-Aware Model Selection for Open LLMs

## Subtitle

Quantifying Leaderboard Instability Under Quantization and Language Shift

## Abstract Draft

Open LLM leaderboards are commonly used to select models for deployment, yet
they usually report scores under fixed benchmark conditions. Real deployments
often use quantized inference and receive multilingual or code-switched inputs.
We study whether clean leaderboard-style model rankings remain stable under
these deployment constraints. StableSelect evaluates open LLMs across precision
settings and input regimes, measuring not only average score but also top-1
ranking flips, rank correlation, quantization drop, and worst-group degradation.
Our goal is to show that the model with the best clean benchmark score may not
be the most stable deployment choice.

## Main Claim

Clean leaderboard rankings can be unreliable for deployment model selection
because the top-ranked model under clean English BF16 evaluation may not remain
top-ranked under quantized, multilingual, or code-switched settings.

## Contributions

1. Deployment perturbation protocol for evaluating open LLM model-selection
   stability across precision and input regimes.
2. Rank-stability evaluation using top-1 flip rate, rank correlation,
   quantization drop, and worst-group drop.
3. A simple Deployment Stability Score that rewards high average performance
   while penalizing worst-case degradation.

## 1. Introduction

Key points:

- Open LLM leaderboards influence deployment decisions.
- Deployment conditions differ from clean benchmark conditions.
- Quantization and language shift can affect models differently.
- The real risk is not only performance loss; it is wrong model selection.

End the introduction with:

> We ask whether the model selected by clean leaderboard-style evaluation remains
> the best choice under realistic deployment perturbations.

## 2. Method

### Models

Use three open 7B-8B class instruction models:

- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- Aya-Expanse-8B

### Precision Settings

- BF16/FP16 clean baseline
- 4-bit quantized inference

### Task Groups

- English QA
- Multilingual QA
- Code-switched QA

### Baseline Ranking

The baseline ranking is the model ordering under:

```text
English QA + BF16
```

All deployment rankings are compared against this baseline.

## 3. Metrics

Use the formal definitions in `paper/math_equations.md`.

### Average Score

Mean score across evaluated settings.

### Quantization Drop

```text
Quantization Drop = Score(BF16) - Score(4-bit)
```

### Worst-Group Drop

```text
Worst-Group Drop = max drop across evaluated task/language groups
```

### Top-1 Flip

```text
Top-1 Flip = 1 if the top model changes relative to clean baseline
```

### Rank Correlation

Spearman rank correlation between clean baseline ranking and each deployment
ranking.

### Deployment Stability Score

```text
DSS(model) = Average Score Across Settings - Worst-Group Drop
```

This intentionally simple score is easier to defend in a short workshop paper.

The main model-selection mismatch is:

$$
\arg\max_{m \in M} Q(m, s_0)
\ne
\arg\max_{m \in M} \text{DSS}(m)
$$

In words: the clean benchmark winner is not the stability-aware deployment
winner.

## 4. Experiments

Report:

- hardware
- inference framework
- model IDs
- quantization method
- number of examples per task
- decoding settings
- exact benchmark/task names

## 5. Results

Required tables and figures:

```text
Table 1: Experimental setup
Table 2: Scores by model, precision, and task group
Table 3: Rank stability and top-1 flips
Figure 1: Ranking flip heatmap
Figure 2: Average score vs Deployment Stability Score
```

Main result questions:

- Does the clean baseline winner remain the winner?
- Which deployment settings cause ranking flips?
- Which model has the largest worst-group drop?
- Which model has the highest Deployment Stability Score?

## 6. Discussion

Interpret the result as model-selection risk:

- leaderboard averages can hide deployment instability,
- quantization can change which model is preferable,
- multilingual and code-switched inputs can expose uneven degradation,
- deployment-aware leaderboards should report stability metrics.

## 7. Limitations

- small number of models,
- benchmark subsets,
- one quantization method,
- limited language coverage,
- results should not be treated as a universal leaderboard.

## 8. Conclusion

StableSelect argues that deployment model selection should consider stability,
not only clean average benchmark score. A model that wins the clean benchmark
may be less reliable once compressed or evaluated under language shift.
