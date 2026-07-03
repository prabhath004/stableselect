# Reviewer Story

## One-Sentence Story

StableSelect shows that clean leaderboard rankings can be an unreliable proxy
for deployment model selection because rankings may change under quantization
and language shift.

## What Reviewers Need to Believe

1. People use leaderboards to choose deployment models.
2. Deployment settings differ from clean leaderboard settings.
3. These differences can change the model ranking, not only lower scores.
4. Ranking instability is measurable with simple, reproducible metrics.
5. Stability-aware reporting would improve model selection.

## What Would Make the Paper Weak

- Only showing that every model loses some score after quantization.
- Too many tasks without a clear model-selection claim.
- A complicated score that reviewers cannot interpret.
- No raw outputs or reproducibility path.

## What Would Make the Paper Strong

- A clear top-1 ranking flip.
- A model with high clean score but poor worst-group degradation.
- A model with lower clean score but better Deployment Stability Score.
- A simple figure that makes the model-selection mismatch obvious.

