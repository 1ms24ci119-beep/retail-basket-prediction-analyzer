# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/2xr5tgxb43/badge.svg)](https://ask.thili.ai/b/2xr5tgxb43)

This project, the Retail Basket Prediction Analyzer, helps retail analysts compare next-item prediction accuracy. It uses both an n-gram model and a small language model, trained on simulated shopping baskets, to understand which model performs better in predicting future purchases.

## What you are building

**Goal.** You will build a Retail Basket Prediction Analyzer that compares next-item prediction accuracy between an n-gram model and a small language model, both trained on simulated shopping baskets data.

**Done when.** The project is complete when you have trained both an n-gram model and a small language model on simulated retail basket data, evaluated their next-item prediction performance using relevant metrics, and presented a clear comparison of their accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Generate a synthetic dataset of shopping baskets with planted itemsets.**  
      Hint: Use the data generation techniques from the 'Teach a Language Model What Sells Together' drill.
- [ ] **2. Implement the n-gram model to predict next items in a basket.**  
      Hint: Apply the n-gram model building and prediction methods from the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Pretrain a small language model (like in the drill) on the basket sequences.**  
      Hint: Adapt the small language model training process from the 'Teach a Language Model What Sells Together' drill.
- [ ] **4. Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).**  
      Hint: Implement standard next-item prediction metrics like accuracy and rank, similar to how model performance is considered in the drills.
- [ ] **5. Present a comparison of the models' performance.**  
      Hint: Summarize and visualize the evaluation results for both models to highlight their comparative performance.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Analyze specific itemsets where one model performs significantly better or worse than the other.
- Visualize prediction confidence or specific error patterns for both models.
- Experiment with different hyperparameters for the small language model or n-gram model to observe performance changes.
