# CourtIQ

**An end-to-end machine learning platform for ATP tennis prediction, probabilistic calibration, and market evaluation.**

CourtIQ is an independent sports analytics research project that combines machine learning, custom rating systems, feature engineering, and probabilistic modeling to evaluate ATP Tour tennis matches.

Rather than simply predicting winners, CourtIQ focuses on estimating match probabilities, comparing those probabilities to efficient betting markets, and identifying situations where statistically meaningful pricing discrepancies may exist.

The project serves as both an ongoing research platform and a transparent public validation of its methodology through timestamped predictions and forward-tested performance.

Live predictions are published at [meliasean.github.io/CourtIQ](https://meliasean.github.io/CourtIQ).

---

# At a Glance

* Developed as an independent applied machine learning research project
* 1,400+ ATP Tour matches used for model development and validation
* ATP Tour-level tournaments across multiple surfaces
* Random Forest prediction model with probability calibration
* Custom surface-adjusted Elo rating system
* Market-aware probability evaluation against consensus bookmaker pricing
* Proprietary decision engine for identifying potential value opportunities
* Live, timestamped predictions tracked before match start
* Ongoing forward validation throughout the 2026–2027 ATP seasons

---

# Project Overview

CourtIQ consists of two independent analytical layers.

## 1. Match Prediction Engine

The prediction engine generates calibrated win probabilities for every tracked ATP Tour match.

Predictions are produced using a Random Forest classifier trained on historical ATP match data spanning multiple seasons. The model incorporates a broad collection of player and match characteristics, including:

* Surface-adjusted Elo ratings
* Recent player form
* Surface performance
* Head-to-head history
* Ranking-based features
* Match context
* Additional engineered statistical features

Raw model outputs are calibrated using isotonic regression to improve probability estimates and reduce known prediction bias.

The objective of this model is not simply classification accuracy, but producing well-calibrated probability estimates suitable for downstream decision-making.

---

## 2. Decision Engine

The second layer evaluates whether predicted probabilities appear meaningfully different from market expectations.

Rather than recommending every model prediction, CourtIQ applies a proprietary multi-factor decision framework that evaluates:

* Model confidence
* Market-implied probability
* Agreement between independent predictive signals
* Volatility and uncertainty
* Additional quantitative factors

Most matches are intentionally excluded.

Only matches meeting predefined statistical criteria are assigned to one of several conviction buckets for ongoing evaluation.

This philosophy reflects the belief that accurate prediction alone does not necessarily create actionable decisions.

---

# System Architecture

```
Historical ATP Data
          │
          ▼
Data Cleaning & Feature Engineering
          │
          ▼
Custom Surface-Adjusted Elo System
          │
          ▼
Random Forest Prediction Model
          │
          ▼
Probability Calibration
          │
          ▼
Market Probability Comparison
          │
          ▼
Proprietary Decision Engine
          │
          ▼
Public Prediction Website
```

---

# Validation Philosophy

CourtIQ emphasizes transparent evaluation over retrospective optimization.

Every tracked prediction is published before the match begins.

Results are recorded after completion using official match outcomes.

No historical predictions are removed or selectively reported.

Additionally:

* The decision engine remains fixed during each live validation period.
* Mid-stream parameter adjustments are intentionally avoided.
* Performance is measured under live conditions rather than retrospective simulations.

The objective is to measure how the methodology performs prospectively rather than optimize results after outcomes are already known.

---

# Current Validation

Ongoing evaluation has focused on measuring performance relative to efficient betting markets rather than prediction accuracy alone.

Current observations include:

* The prediction model performs comparably to or modestly better than consensus bookmaker probabilities across evaluated ATP matches.
* The proprietary decision engine has produced positive performance versus consensus bookmaker pricing across an accumulating sample of qualifying selections combining historical backtest and current live evaluation. Live results are published in full on the site linked above.
* Analysis of matches classified as **Avoid** illustrates why prediction accuracy alone is insufficient: heavy market favorites frequently win but produce negative expected return once bookmaker margins are considered.

These results continue to be monitored through live forward validation rather than retrospective optimization.

---

# Public Demonstration

The public website publishes:

* Match predictions
* Value selections
* Conviction buckets
* Timestamped release before match start
* Historical tracking of results

The purpose of the site is to provide a transparent record of model behavior under real-world conditions.

---

# Proprietary Components

While the overall modeling framework and research methodology are documented publicly, several implementation details remain private.

These include, but are not limited to:

* Decision engine methodology
* Feature weighting strategies
* Match scoring formulas
* Selection thresholds
* Override logic
* Calibration refinements
* Model tuning procedures
* Certain proprietary feature engineering techniques

These components represent ongoing independent research and potential future commercial intellectual property.

---

# Project Status

CourtIQ is currently an independent research initiative.

The project is not presently operated as a commercial product, subscription service, or paid betting platform.

The primary objective during the 2026–2027 ATP seasons is continued forward validation, refinement, and statistical evaluation under live conditions before any future commercialization decisions are made.

---

# Technology

* Python
* scikit-learn
* Pandas
* NumPy
* Random Forest
* Isotonic Regression
* ATP historical match database
* Custom Elo implementation
* GitHub Pages

---

# License

**© 2026 Sean Melia. All rights reserved.**

This repository is provided for demonstration and portfolio purposes only.

No portion of this repository may be copied, modified, redistributed, reproduced, reverse engineered, or used for commercial purposes without the express written permission of the author.

The public documentation is intended to describe the project at a high level while protecting proprietary implementation details.

---

# Disclaimer

CourtIQ is an independent research project.

Nothing contained within this repository or the associated website constitutes financial, wagering, or investment advice.

All published predictions represent the output of statistical models and are inherently uncertain. Any decisions made using this information are solely the responsibility of the end user.
