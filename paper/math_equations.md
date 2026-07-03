# StableSelect Math Equations

Use these equations in the method or metrics section of the paper.

## Notation

Let:

```text
M = set of evaluated models
S = set of deployment settings
G = set of task or language groups
```

For a model `m`, setting `s`, and group `g`, define:

$$
Q(m, s, g)
$$

as the measured task score, such as accuracy, exact match, F1, or a
benchmark-native score.

The clean leaderboard-style baseline setting is:

$$
s_0 = (\text{English}, \text{BF16})
$$

## Average Deployment Score

The average score of model `m` across deployment settings is:

$$
\text{AvgScore}(m) =
\frac{1}{|S|}
\sum_{s \in S} Q(m, s)
$$

If each setting contains multiple groups:

$$
\text{AvgScore}(m) =
\frac{1}{|S||G|}
\sum_{s \in S}
\sum_{g \in G}
Q(m, s, g)
$$

## Quantization Drop

For task group `g`, the 4-bit quantization drop is:

$$
\Delta_{\text{quant}}(m, g) =
Q(m, \text{BF16}, g) -
Q(m, \text{4bit}, g)
$$

The model-level quantization drop is:

$$
\Delta_{\text{quant}}(m) =
\frac{1}{|G|}
\sum_{g \in G}
\left[
Q(m, \text{BF16}, g) -
Q(m, \text{4bit}, g)
\right]
$$

## Worst-Group Drop

Worst-group drop measures the largest degradation for any task or language
group compared with the clean baseline:

$$
\Delta_{\text{worst}}(m) =
\max_{s \in S, g \in G}
\left[
Q(m, s_0, g) - Q(m, s, g)
\right]
$$

If the clean baseline exists only for English, use:

$$
\Delta_{\text{worst}}(m) =
\max_{s \in S, g \in G}
\left[
Q(m, \text{English}, \text{BF16}) - Q(m, s, g)
\right]
$$

## Code-Switch Penalty

The code-switch penalty compares a model's monolingual score with its
code-switched score:

$$
\Delta_{\text{cs}}(m) =
Q(m, \text{monolingual}) -
Q(m, \text{code-switched})
$$

For precision-specific analysis:

$$
\Delta_{\text{cs}}(m, p) =
Q(m, p, \text{monolingual}) -
Q(m, p, \text{code-switched})
$$

where `p` is a precision setting such as BF16 or 4-bit.

## Ranking Function

Let `R_s(m)` be the rank of model `m` under setting `s`, where rank 1 is best:

$$
R_s(m) =
1 + \left|
\{m' \in M : Q(m', s) > Q(m, s)\}
\right|
$$

The clean baseline ranking is:

$$
R_{s_0}(m)
$$

## Top-1 Flip

Let the top model under setting `s` be:

$$
m_s^* = \arg\max_{m \in M} Q(m, s)
$$

The clean baseline winner is:

$$
m_0^* = \arg\max_{m \in M} Q(m, s_0)
$$

The top-1 flip indicator is:

$$
\text{Flip}(s) =
\mathbb{1}[m_s^* \ne m_0^*]
$$

The top-1 flip rate across deployment settings is:

$$
\text{FlipRate} =
\frac{1}{|S|-1}
\sum_{s \in S \setminus \{s_0\}}
\mathbb{1}[m_s^* \ne m_0^*]
$$

## Spearman Rank Correlation

For `n` models, Spearman rank correlation between the clean baseline ranking
and deployment setting `s` is:

$$
\rho_s =
1 -
\frac{
6 \sum_{m \in M}
\left(R_{s_0}(m) - R_s(m)\right)^2
}{
n(n^2 - 1)
}
$$

Low `rho_s` means the deployment ranking is unstable relative to the clean
baseline ranking.

## Rank Instability

A simple rank-instability score for setting `s` is:

$$
\text{RankInstability}(s) = 1 - \rho_s
$$

The average rank instability across deployment settings is:

$$
\text{RankInstability} =
\frac{1}{|S|-1}
\sum_{s \in S \setminus \{s_0\}}
(1 - \rho_s)
$$

## Deployment Stability Score

The simple workshop-friendly Deployment Stability Score is:

$$
\text{DSS}(m) =
\text{AvgScore}(m) -
\Delta_{\text{worst}}(m)
$$

This is the recommended version for the first submission because it is simple
and easy to interpret.

## Extended Deployment Stability Score

If the paper needs a more general score, use:

$$
\text{DSS}_{\lambda}(m) =
\text{AvgScore}(m)
- \lambda_1 \Delta_{\text{worst}}(m)
- \lambda_2 \Delta_{\text{quant}}(m)
- \lambda_3 \Delta_{\text{cs}}(m)
$$

where `lambda_1`, `lambda_2`, and `lambda_3` control how strongly the score
penalizes worst-case degradation, quantization degradation, and code-switching
degradation.

For the workshop paper, prefer the simpler DSS unless the extended version is
clearly needed.

## Model-Selection Mismatch

The clean benchmark selection is:

$$
m_{\text{clean}} =
\arg\max_{m \in M} Q(m, s_0)
$$

The stability-aware selection is:

$$
m_{\text{stable}} =
\arg\max_{m \in M} \text{DSS}(m)
$$

StableSelect finds a model-selection mismatch when:

$$
m_{\text{clean}} \ne m_{\text{stable}}
$$

This is the strongest paper result: the model selected by the clean benchmark
is not the model selected by deployment stability.

