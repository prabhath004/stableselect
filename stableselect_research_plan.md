# StableSelect: Deployment-Aware Model Selection for Open LLMs

## Subtitle

**Quantifying Leaderboard Instability Under Quantization, Long Context, and Language Shift**

---

## 1. One-Line Idea

Open LLM leaderboards are commonly used to choose models for deployment, but real deployments often use quantized inference, long-context inputs, and multilingual or code-switched user queries. **StableSelect** studies whether the model that looks best on a clean benchmark is still the best model to deploy under these realistic constraints.

---

## 2. Core Research Question

**Are open LLM leaderboards reliable for choosing models under real deployment constraints?**

More formal version:

> To what extent do deployment constraints—quantization, long-context inputs, and language shift—alter open LLM rankings and expose failures hidden by average benchmark scores?

---

## 3. Why This Is Not Obvious

The weak/obvious idea would be:

> Models perform differently under different conditions.

That is not enough for a research paper.

The stronger research idea is:

> Benchmark-based model selection can become unreliable because the top-ranked model under clean benchmark settings may not be the most stable or safest model under deployment settings.

The paper is not just about performance dropping. It is about **model-selection risk**.

A model can be:

- highest scoring on clean English FP16 benchmarks,
- but unstable under 4-bit quantization,
- weak on low-resource or code-switched inputs,
- brittle on long-context tasks,
- and therefore not the best deployment choice.

---

## 4. Final Paper Title Options

### Preferred Title

**StableSelect: Deployment-Aware Model Selection for Open LLMs**

### Alternate Titles

1. **When Rankings Flip: Evaluating Open LLM Leaderboards Under Deployment Constraints**
2. **Beyond Average Accuracy: Deployment-Stable Evaluation of Open LLMs**
3. **Safe Model Selection Under Quantization, Long Context, and Language Shift**
4. **Are Open LLM Leaderboards Deployment-Reliable?**

---

## 5. Abstract Draft

Open LLM leaderboards are commonly used to select models for deployment, yet they usually report performance under fixed benchmark conditions. Real deployments often use quantized inference, long-context inputs, and multilingual or code-switched user queries. We study whether leaderboard rankings remain stable under these deployment constraints. Evaluating open LLMs across precision levels, context regimes, and language settings, we find that average benchmark scores can hide large worst-group degradation and that model rankings can change under realistic deployment perturbations. We introduce rank-stability metrics and a Deployment Stability Score to quantify when a model is not only accurate, but reliable to deploy. Our results suggest that model-selection decisions should account for deployment stability rather than relying on average benchmark scores alone.

---

## 6. Main Claim

The main claim we want to support is:

> The highest-scoring model under standard benchmark settings is often not the safest deployment choice. Model rankings can flip under deployment perturbations, and stability-aware metrics are needed to identify models that remain reliable across realistic settings.

A strong final result would look like:

> Across 3 open LLMs and 24–36 deployment settings, the top-ranked model changed in a substantial fraction of settings. 4-bit quantization caused only modest average English degradation but produced much larger degradation on low-resource or code-switched inputs. The model with the highest average benchmark score was not always the model with the highest Deployment Stability Score.

---

## 7. Contributions

The paper should make exactly three contributions:

1. **Deployment Perturbation Protocol**  
   We introduce a controlled protocol for evaluating open LLMs across precision, context length, and language shift.

2. **Rank-Stability Evaluation**  
   We quantify leaderboard instability using metrics such as top-1 flip rate, Kendall tau rank correlation, Spearman rank correlation, and worst-group degradation.

3. **Deployment Stability Score**  
   We propose a simple stability-aware score that helps identify models that are robust under realistic deployment constraints, not only models that win average benchmark scores.

---

## 8. Experimental Design

### 8.1 Models

Minimum set:

1. **Llama-3.1-8B**
2. **Qwen2.5-7B**
3. **Aya-Expanse-8B**

Optional fourth model if compute allows:

4. **Mistral-7B / Gemma-7B / another strong open 7B–9B model**

### 8.2 Precision Settings

Evaluate each model under:

1. **FP16** — high-quality baseline
2. **8-bit quantization** — practical lower-cost inference
3. **4-bit quantization** — aggressive low-memory inference

### 8.3 Input Regimes

Evaluate each model under:

1. **English short-context tasks**
2. **Multilingual short-context tasks**
3. **Code-switched tasks**
4. **Long-context QA tasks**

Minimum viable version:

```text
3 models × 3 precisions × 3 task regimes = 27 settings
```

Full version:

```text
3 models × 3 precisions × 4 task regimes = 36 settings
```

---

## 9. Benchmarks and Data

### English Baseline

Use a small subset of common LLM evaluation tasks, such as:

- MMLU-style QA subset
- commonsense QA subset
- reasoning QA subset

### Multilingual Evaluation

Use a multilingual benchmark subset with high-, medium-, and low-resource languages.

Candidate languages:

- English
- Hindi
- Telugu or Tamil
- Arabic
- Spanish
- Swahili or Bengali

The point is not to cover every language. The point is to test whether deployment constraints hurt languages unequally.

### Code-Switched Evaluation

Create or use a compact code-switched test set.

Example prompts:

```text
English: Explain database indexing in simple terms.
Code-switched: Bro database indexing ante enti? Explain simply.
Hindi-English: Database indexing kya hota hai? Simple example ke saath batao.
```

### Long-Context Evaluation

Use a small long-context QA subset where the model must retrieve answers from long documents.

Possible tasks:

- long-document QA
- synthetic needle-in-a-haystack-style retrieval
- long legal/report-style QA
- long context benchmark subset

---

## 10. Metrics

### 10.1 Standard Task Metrics

- Accuracy
- Exact match
- F1
- Benchmark-native score

### 10.2 Deployment Metrics

- Latency
- Tokens per second
- Peak memory usage
- Cost-normalized score, if feasible

### 10.3 Stability Metrics

#### Top-1 Flip Rate

Measures how often the best model changes across settings.

```text
Top-1 Flip Rate = Number of settings where top model changes / Total settings
```

#### Kendall Tau / Spearman Rank Correlation

Measures how similar model rankings are between clean benchmark settings and deployment settings.

Example:

```text
Ranking under English FP16:
1. Llama
2. Qwen
3. Aya

Ranking under multilingual 4-bit:
1. Aya
2. Qwen
3. Llama
```

Low correlation means the leaderboard is unstable.

#### Worst-Group Drop

Measures the largest performance drop for any language or subgroup.

```text
Worst-Group Drop = max drop across evaluated languages/groups
```

#### Quantization Degradation

Measures how much performance drops from FP16 to int8/int4.

```text
Quantization Drop = Score(FP16) - Score(int4)
```

#### Code-Switch Penalty

Measures how much performance drops from monolingual to code-switched prompts.

```text
Code-Switch Penalty = Score(monolingual) - Score(code-switched)
```

---

## 11. Deployment Stability Score

A simple score:

```text
Deployment Stability Score = Average Score
                           - λ1(Rank Instability)
                           - λ2(Worst-Group Drop)
                           - λ3(Quantization Drop)
```

Plain English:

> A good deployment model should not only score high. It should remain stable when compressed, tested on long inputs, and used across languages.

The exact formula can be adjusted after pilot experiments.

Simpler non-parametric version:

```text
DSS(model) = Average Score Across Deployment Settings - Worst-Case Drop
```

This may be easier for a 5–6 page workshop paper.

---

## 12. Expected Findings

The paper becomes strong if it finds one or more of these:

1. The top-ranked model changes under quantization.
2. 4-bit quantization hurts low-resource/code-switched inputs more than English.
3. Long-context tasks cause larger ranking instability than short-context tasks.
4. The highest average-scoring model is not the most stable model.
5. Average benchmark scores hide serious worst-group failures.

Example final claim:

> We find that clean leaderboard rankings are only partially predictive of deployment performance. The top model changes across multiple quantized and multilingual settings, and the model with the best average score is not always the most stable deployment choice.

---

## 13. Paper Structure for 5–6 Page Submission

### Page 1: Abstract + Introduction

Explain:

- Open LLM leaderboards influence model selection.
- Real deployments use quantization, long context, and multilingual prompts.
- Fixed benchmark scores may not predict deployment performance.
- Research question: Are leaderboard rankings stable under deployment constraints?

End with contributions.

### Page 2: Method

Describe:

- deployment perturbation matrix,
- precision settings,
- input regimes,
- rank-stability metrics,
- Deployment Stability Score.

### Page 3: Experimental Setup

Include:

- models,
- benchmarks,
- hardware,
- inference framework,
- quantization method,
- evaluation procedure.

### Page 4: Results

Include:

- raw performance table,
- ranking flip table,
- worst-language degradation table,
- heatmap showing which model wins under each deployment setting.

### Page 5: Discussion

Explain:

- leaderboard rankings can be unstable,
- average accuracy hides worst-case degradation,
- some models are high-scoring but brittle,
- deployment-aware leaderboards should report stability.

### Page 6: Conclusion + Limitations

Limitations:

- small number of models,
- benchmark subsets due to compute constraints,
- results may differ for larger models,
- not a universal leaderboard,
- goal is to show that leaderboard stability should be measured.

---

## 14. Figure Ideas

### Figure 1: StableSelect Evaluation Pipeline

```text
Open LLMs
   ↓
Precision Settings: FP16 / int8 / int4
   ↓
Input Regimes: English / Multilingual / Code-switched / Long-context
   ↓
Scores + Latency + Memory
   ↓
Rank-Stability Metrics
   ↓
Deployment Stability Score
```

### Figure 2: Ranking Flip Heatmap

Rows:

- models

Columns:

- deployment settings

Color:

- winning model or score

Purpose:

- visually show that the winning model changes across settings.

### Figure 3: Average Score vs Stability Score

X-axis:

- average benchmark score

Y-axis:

- Deployment Stability Score

Purpose:

- show that the highest average model may not be the most stable model.

---

## 15. Workshop Positioning

The accepted NeurIPS 2026 workshop list should be checked once public. Until then, target these workshop categories:

1. **LLM Evaluation / Foundation Model Evaluation**
2. **ML for Systems / Efficient ML**
3. **Reliable ML / Trustworthy ML**
4. **Multilingual / Low-Resource LLMs**
5. **Responsible Foundation Models**
6. **Quantization / Model Compression**
7. **Long-Context LLMs / RAG Evaluation**

### Best Primary Target

**LLM Evaluation / Foundation Model Evaluation**

Positioning:

> This paper studies whether open LLM leaderboard rankings remain stable under deployment constraints.

### Best Backup Target

**ML for Systems / Efficient ML**

Positioning:

> This paper studies quantized LLM deployment and shows why model selection must account for stability, latency, memory, and worst-case degradation.

### Multilingual Workshop Version

Title:

**Do Quantized LLMs Fail Unequally Across Languages?**

Positioning:

> This paper shows that quantization and deployment constraints can amplify multilingual and code-switched failures.

### Reliable ML Version

Title:

**Leaderboard Instability as a Reliability Risk in Open LLM Deployment**

Positioning:

> This paper frames model ranking instability as a reliability problem under deployment distribution shift.

---

## 16. Implementation Plan

### Week 1: Scope Lock and Setup

- Choose final 3 models.
- Choose benchmarks/subsets.
- Set up inference framework.
- Validate FP16 inference.

Deliverable:

- experiment config file and baseline run.

### Week 2: Quantization Setup

- Add int8 inference.
- Add int4 inference.
- Validate memory and latency logging.

Deliverable:

- precision comparison sanity report.

### Week 3: Benchmark Loaders

- Implement English loader.
- Implement multilingual loader.
- Implement code-switched loader.
- Implement long-context loader.

Deliverable:

- unified evaluation harness.

### Week 4: Pilot Runs

- Run 5–10% of the full matrix.
- Check if rankings move.
- Drop tasks that are too noisy or expensive.

Deliverable:

- pilot table and go/no-go decision.

### Weeks 5–6: Main Experiments

- Run full evaluation matrix.
- Save raw outputs.
- Track latency/memory/tokens per second.

Deliverable:

- complete raw results.

### Week 7: Analysis

- Compute rank stability.
- Compute top-1 flip rate.
- Compute worst-language drop.
- Compute Deployment Stability Score.
- Make heatmaps and tables.

Deliverable:

- final figures.

### Week 8: Paper Writing

- Write 5–6 page paper.
- Prepare appendix if allowed.
- Clean repo and README.
- Submit to best-fit workshop.

Deliverable:

- submission-ready PDF and GitHub repo.

---

## 17. Risks and Mitigations

### Risk 1: Scope too large

Mitigation:

- Use 3 models only.
- Use benchmark subsets.
- Remove code-switched or long-context if needed.

### Risk 2: Results are not dramatic

Mitigation:

- Focus on worst-group degradation and rank stability, not only average accuracy.
- Even small average drops can hide large subgroup failures.

### Risk 3: Quantization setup is painful

Mitigation:

- Use one stable quantization library first.
- Start with bitsandbytes int8/int4.
- Add AWQ/GPTQ only if needed.

### Risk 4: Long-context evaluation is too expensive

Mitigation:

- Use a small subset.
- Use shorter “long” contexts first, such as 8k–16k tokens.
- Make long-context optional if compute becomes tight.

---

## 18. What Makes This Workshop-Acceptable

This is a strong workshop-style paper because it is:

- practical,
- empirical,
- reproducible,
- evaluation-focused,
- deployment-relevant,
- feasible for a solo author,
- connected to current LLM evaluation and systems concerns.

The paper does not need a new model. Its value comes from:

- better experimental design,
- stability metrics,
- reproducible evaluation harness,
- clear evidence that average leaderboard scores are incomplete.

---

## 19. Final Submission Pitch

Use this when describing the paper:

> StableSelect studies whether open LLM leaderboards are reliable for deployment model selection. We evaluate open LLMs under quantized inference, long-context inputs, and multilingual/code-switched prompts. Instead of only reporting average accuracy, we measure ranking flips, worst-group degradation, and deployment stability. Our results show that the model with the best clean benchmark score may not be the safest or most stable model to deploy.

---

## 20. Final Decision

Proceed with:

# StableSelect: Deployment-Aware Model Selection for Open LLMs

Core angle:

> Leaderboards tell us which model wins under clean benchmark settings. StableSelect tests whether that model is still the right deployment choice under quantization, long context, and language shift.

Primary workshop target:

> LLM Evaluation / Foundation Model Evaluation

Backup workshop target:

> ML for Systems / Efficient ML

Paper length:

> 5–6 pages

Submission type:

> Empirical workshop paper with reproducible benchmark harness and stability metrics.
