# Figure Plan

## Figure 1: StableSelect Pipeline

Purpose: explain the experimental protocol.

```text
Models
  -> Precision settings
  -> Task groups
  -> Scores
  -> Rank-stability metrics
  -> Deployment Stability Score
```

Use in the method section.

## Figure 2: Ranking Flip Heatmap

Purpose: show whether the winning model changes across settings.

Rows:

- task groups
- precision settings

Columns:

- top model or model rank

Color:

- winning model
- or rank position

Use in the main results section.

## Figure 3: Average Score vs Deployment Stability Score

Purpose: show whether average benchmark performance and deployment stability
select the same model.

X-axis:

- average score

Y-axis:

- Deployment Stability Score

Each point:

- one model

Use in the discussion section.

