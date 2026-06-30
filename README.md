# Hillstrom Email Campaign Uplift Modeling

Causal machine learning project for estimating email campaign uplift and improving customer targeting using the Hillstrom MineThatData email marketing experiment.

## Repository Description

This project compares traditional response modeling with uplift modeling and causal machine learning methods. The goal is not only to predict which customers are likely to visit, convert, or spend, but to estimate which customers are likely to change their behavior **because of** receiving a campaign email.

The main business question is:

> Which customers should receive a campaign email because the email is expected to create incremental value?

This is a causal decision-making problem. A high predicted probability of response does not necessarily mean a high treatment effect. Some customers may visit or purchase with or without an email. Others may respond only when contacted. The project focuses on identifying the customers with the highest expected incremental response.

---

## Dataset

The project uses the Hillstrom email marketing dataset from Kevin Hillstrom's MineThatData email analytics challenge. The dataset contains 64,000 customer-level observations from a randomized email experiment.

Customers were assigned to one of three groups:

| Group | Description |
|---|---|
| `Mens E-Mail` | Customer received an email featuring men's merchandise |
| `Womens E-Mail` | Customer received an email featuring women's merchandise |
| `No E-Mail` | Customer did not receive a campaign email and serves as the control group |

The dataset includes historical customer behavior, campaign assignment, and post-campaign outcomes.

### Key Variables

| Category | Variables | Description |
|---|---|---|
| Customer history | `recency`, `history`, `history_segment` | How recently and how much the customer purchased historically |
| Product behavior | `mens`, `womens` | Whether the customer previously purchased men's or women's merchandise |
| Customer type | `newbie` | Whether the customer is a newer customer |
| Geography / channel | `zip_code`, `channel` | Customer zip-code type and prior purchase channel |
| Treatment | `segment` | Email assignment: men's email, women's email, or no email |
| Outcomes | `visit`, `conversion`, `spend` | Post-campaign visit, purchase conversion, and spend |

The main outcome for modeling is `visit`, because visits are more frequent than purchases and provide a stronger signal. Secondary outcomes include `conversion` and `spend`.

---

## Why This Is Not Just Classification

A traditional response model estimates:

```text
P(Y = 1 | X)
```

This answers:

> Who is likely to visit, convert, or buy?

An uplift model estimates:

```text
P(Y = 1 | X, T = 1) - P(Y = 1 | X, T = 0)
```

This answers:

> Who is more likely to visit, convert, or buy because of the campaign?

The project will compare ordinary response ranking against uplift-based targeting to show why high response probability and high incremental response are not the same thing.

---

## Causal Setup

The first modeling setup collapses the two email groups into a binary treatment:

| Treatment Indicator | Meaning |
|---|---|
| `T = 1` | Customer received either the men's email or the women's email |
| `T = 0` | Customer received no email |

For a binary outcome `Y`, customer features `X`, and treatment indicator `T`, the target estimand is the conditional average treatment effect, or expected uplift:

```text
tau(X) = E[Y(1) - Y(0) | X]
```

For classification-style outcomes such as `visit`, this is estimated as:

```text
tau(X) = P(Y = 1 | X, T = 1) - P(Y = 1 | X, T = 0)
```

The project will not claim to observe the true individual treatment effect for each customer. For each customer, we only observe what happened under the treatment condition they actually received. The goal is to estimate expected uplift for customers with similar observed features.

---

## Recommended Implementation Plan

The project should include more than a single uplift score. The final implementation should compare several methods that estimate the same general target, `tau(X)`, but use different modeling assumptions.

### 1. Exploratory Data Analysis and Experiment Validation

Before modeling, verify that the experiment is usable for causal/uplift analysis.

Required checks:

- Number of customers in each treatment group
- Visit, conversion, and spend rates by treatment group
- Treatment-control balance across pre-treatment features
- Standardized mean differences for numeric and binary features
- Distribution of `recency`, `history`, `channel`, `zip_code`, `mens`, `womens`, and `newbie` by treatment group
- Segment-level outcome rates and lift

Important rule:

> Only use pre-treatment customer attributes as features. Do not use `visit`, `conversion`, or `spend` as predictors.

### 2. Average Treatment Effect Baselines

Start with simple treatment-control comparisons.

For binary outcomes:

```text
ATE = E[Y | T = 1] - E[Y | T = 0]
```

For spend:

```text
ATE_spend = E[spend | T = 1] - E[spend | T = 0]
```

This gives a campaign-level baseline before estimating customer-level heterogeneity.

Recommended comparisons:

- Any email vs no email
- Men's email vs no email
- Women's email vs no email
- Men's email vs women's email, for descriptive comparison
- Segment-level ATEs by customer history, channel, product behavior, and newbie status

### 3. Traditional Response Models

Build regular supervised learning models as a baseline.

Target examples:

```text
P(visit = 1 | X)
P(conversion = 1 | X)
E[spend | X]
```

Candidate models:

- Logistic regression
- Decision tree
- Random forest
- Gradient boosting
- XGBoost or LightGBM, if allowed

These models are useful, but they answer the wrong targeting question by themselves. They predict who is likely to respond, not who is likely to respond because of the email.

### 4. CausalML Meta-Learners

The implementation should include CausalML meta-learners as the main causal machine learning comparison layer.

Recommended CausalML methods:

| Method | Recommended Status | Purpose |
|---|---:|---|
| S-learner | Required | Single model with treatment as a feature |
| T-learner | Required | Separate treated and control outcome models |
| X-learner | Strong optional | Useful comparison method; especially helpful when treatment/control sizes differ |
| R-learner | Optional / advanced | Residualized treatment-effect learner; more advanced and may be overkill |
| DR-learner | Optional robustness | Doubly robust style comparison; useful but not necessary as the main method |
| Uplift tree / causal tree | Recommended | Directly interpretable uplift segmentation method |
| Uplift random forest | Optional | Ensemble version of direct uplift trees |

Suggested imports may look like this, depending on the installed CausalML version:

```python
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.metrics import plot_gain, plot_qini, auuc_score, qini_score
from causalml.propensity import ElasticNetPropensityModel
```

If exact class names differ by package version, check the installed CausalML documentation or inspect the package with:

```python
import causalml
import causalml.inference.meta as meta
import causalml.inference.tree as tree

dir(meta)
dir(tree)
```

---

## Propensity Score Strategy

Because the Hillstrom campaign is randomized, propensity scores are not needed to correct the kind of selection bias that appears in observational data. However, some CausalML learners still use or accept a propensity score as part of their implementation.

### Binary Any-Email Setup

If the treatment is collapsed into:

```text
T = 1 for Mens E-Mail or Womens E-Mail
T = 0 for No E-Mail
```

then the known experimental propensity is approximately:

```text
p = P(T = 1) ≈ 2/3
```

because two of the three randomized groups received an email.

### Pairwise Email-vs-Control Setup

If comparing one email type against control, for example:

```text
Mens E-Mail vs No E-Mail
Womens E-Mail vs No E-Mail
```

then the analysis subset has two arms, so the known propensity is approximately:

```text
p = P(T = 1) ≈ 1/2
```

### How to Use Propensity in This Project

Use the following practical rule:

| Method | Propensity Recommendation |
|---|---|
| S-learner | Do not rely on propensity as the main correction; treatment is included as a model input |
| T-learner | Do not rely on propensity as the main correction; models are fit separately by treatment arm |
| X-learner | Use known randomized propensity when possible, or CausalML's propensity model for comparison |
| R-learner | Requires propensity; use known randomized propensity when possible |
| DR-learner | Requires propensity; use known randomized propensity when possible |
| IPW-only methods | Not a priority because assignment is randomized |

If using CausalML's `ElasticNetPropensityModel`, treat it as an implementation convenience or robustness check, not as the reason the result is causal. In a randomized experiment, the estimated propensity should be close to the treatment assignment rate and should not show strong predictable structure from customer features.

Recommended diagnostic:

- Fit the propensity model.
- Plot predicted propensity distribution.
- Confirm it is approximately centered around the randomized assignment probability.
- If the propensity model finds strong signal, investigate whether the treatment encoding, filtering, or data processing is wrong.

---

## Detailed Model Implementation

### T-Learner / Two-Model Approach

The T-learner trains one outcome model on treated customers and another outcome model on control customers.

```text
m1(X) = P(Y = 1 | X, T = 1)
m0(X) = P(Y = 1 | X, T = 0)
tau_hat(X) = m1(X) - m0(X)
```

Recommended base learners:

- Logistic regression for an interpretable baseline
- Random forest for nonlinear patterns
- Gradient boosting, XGBoost, or LightGBM for stronger predictive performance

Implementation notes:

- Use the same features for the treated and control models.
- Use the same train/test split across methods.
- Predict both potential outcomes for every customer in the test set.
- Rank customers by `tau_hat`.

### S-Learner / Interaction Model

The S-learner fits one model that includes both customer features and treatment status.

```text
m(X, T) = P(Y = 1 | X, T)
tau_hat(X) = m(X, T = 1) - m(X, T = 0)
```

To make the S-learner more useful, include treatment-feature interactions:

```text
visit ~ recency + history + channel + treatment
      + treatment:history
      + treatment:recency
      + treatment:channel
      + treatment:mens
      + treatment:womens
      + treatment:newbie
```

Implementation notes:

- Score each test customer twice: once with `T = 1` and once with `T = 0`.
- The difference is the estimated uplift.
- A simple S-learner without interactions may understate heterogeneous treatment effects, especially if the base model does not naturally capture interactions.

### X-Learner

The X-learner is an extension of the T-learner. It first estimates treated and control outcome models, then imputes treatment effects for each group and learns treatment-effect models.

High-level process:

1. Fit treated outcome model `m1(X)`.
2. Fit control outcome model `m0(X)`.
3. Impute treatment effects:
   - For treated customers: `D1 = Y - m0(X)`
   - For control customers: `D0 = m1(X) - Y`
4. Fit models for the imputed treatment effects.
5. Combine estimates using propensity weights.

Why include it:

- It gives another individualized treatment-effect method beyond S- and T-learners.
- It can perform well when treated and control sample sizes are not equal.
- In the collapsed any-email setup, the treated group is about twice as large as the control group, so X-learner is a useful comparison.

### R-Learner

The R-learner residualizes both the outcome and the treatment, then learns treatment-effect heterogeneity from the residualized relationship.

Conceptually, it uses:

```text
Y - m(X)
T - e(X)
```

where:

- `m(X)` is the expected outcome model
- `e(X)` is the propensity model

Why it is optional:

- It is more advanced than the S-, T-, and X-learners.
- It requires careful nuisance model estimation.
- It may be more than is necessary for the course project, but it is a strong extension if time allows.

### Transformed Outcome Model

The transformed outcome approach creates a pseudo-outcome whose conditional expectation equals the treatment effect.

For binary treatment with propensity `p`:

```text
Y_star = (T * Y / p) - ((1 - T) * Y / (1 - p))
```

Then train a regression model:

```text
E[Y_star | X] ≈ tau(X)
```

Why include it:

- It is a different method family from S/T/X/R meta-learners.
- It directly connects to uplift modeling literature.
- It gives the project another method to satisfy the requirement to compare multiple causal approaches.

Implementation notes:

- Use the known randomized propensity when possible.
- Use regression models, even if the original outcome is binary.
- Compare by ranking performance, not only by raw predicted uplift values.

### Uplift Tree / Causal Tree

An uplift tree directly partitions customers into groups where treatment-control lift differs.

Unlike a normal decision tree, an uplift tree does not split only to predict `Y`. It splits to find segments with different treatment effects.

Why include it:

- It is more interpretable than most meta-learners.
- It can create business-friendly customer segments.
- It can explain which kinds of customers appear persuadable or not worth targeting.

Implementation notes:

- Use minimum treated and control samples per leaf.
- Prune or limit depth to avoid overfitting.
- Prefer honest splitting if available: one sample to choose splits, another to estimate leaf treatment effects.
- Report leaf-level treatment effect, sample size, treated outcome rate, and control outcome rate.

---

## Multi-Treatment Extension

The Hillstrom dataset has three arms, so the project can move beyond the binary question:

> Should we send an email?

and answer:

> Which email, if any, should we send?

Estimate separate potential outcome models:

```text
m_mens(X)    = P(Y = 1 | X, Mens E-Mail)
m_womens(X)  = P(Y = 1 | X, Womens E-Mail)
m_control(X) = P(Y = 1 | X, No E-Mail)
```

Then compute:

```text
uplift_mens(X)   = m_mens(X) - m_control(X)
uplift_womens(X) = m_womens(X) - m_control(X)
```

Decision rule:

```text
If max(uplift_mens, uplift_womens) <= 0: recommend No E-Mail
If uplift_mens > uplift_womens: recommend Mens E-Mail
If uplift_womens > uplift_mens: recommend Womens E-Mail
```

This extension is useful, but the binary any-email setup should be completed first.

---

## Evaluation Plan

Uplift models should not be evaluated only with accuracy, F1, ROC AUC, or classification error. Those metrics evaluate outcome prediction, not treatment-effect ranking.

### Use Standard Metrics Only for Response Models

For ordinary response models, report:

- Accuracy
- ROC AUC
- Precision
- Recall
- F1 score
- Confusion matrix
- Calibration curve

These metrics are useful for the baseline response model, but they are not enough for uplift modeling.

### Uplift-Specific Metrics

For uplift models, evaluate the ranking and targeting strategy.

Required metrics and visuals:

- Uplift-by-decile table
- Cumulative gain curve
- Qini curve
- AUUC, area under the uplift curve
- Qini coefficient or Qini score
- Uplift at top 10%, 20%, and 30%
- Incremental visits/conversions captured by top-ranked customers
- ROI simulation

### Uplift-by-Decile Evaluation

Process:

1. Predict uplift for each test customer.
2. Sort customers from highest to lowest predicted uplift.
3. Split into deciles.
4. For each decile, calculate:
   - Treated customer count
   - Control customer count
   - Treated visit rate
   - Control visit rate
   - Actual lift

Expected pattern:

> A useful uplift model should have higher observed treatment-control lift in the top predicted-uplift deciles than in lower deciles.

### Qini Curve and AUUC

A Qini curve or uplift curve evaluates how much incremental response is captured as the business targets a larger share of customers.

Process:

1. Rank customers by predicted uplift.
2. Target the top `x%` of customers.
3. Estimate cumulative incremental outcomes.
4. Compare against random targeting.

Interpretation:

- Higher AUUC means the model ranks persuadable customers closer to the top.
- A model can have weak classification AUC but still produce a useful uplift ranking.
- A model can have strong classification AUC but still perform poorly for uplift targeting.

### Policy Value and ROI Simulation

The final project should include a business simulation.

Example assumptions:

| Item | Example Value |
|---|---:|
| Email cost | `$0.05` per customer |
| Incentive cost | `$2.00` per customer |
| Profit per conversion | `$50.00` |

Expected incremental profit:

```text
ExpectedProfit(X) = tau_hat(X) * ValueOfConversion - CostOfTreatment
```

Decision rule:

```text
Send email if ExpectedProfit(X) > 0
```

Compare strategies:

1. Send to everyone
2. Send to no one
3. Send to customers with highest predicted response
4. Send to customers with highest predicted uplift
5. Send only when expected incremental profit is positive

This is the most business-relevant evaluation because it translates uplift scores into targeting decisions.

---

## Model Comparability and Calibration

Do not compare uplift models only by the raw magnitude of their predicted uplift scores. Different learners may produce scores on slightly different scales.

Better comparisons:

- Same train/test split
- Same outcome
- Same feature set
- Same treatment definition
- Same evaluation sample
- Same Qini/AUUC calculation
- Same uplift-by-decile procedure
- Same ROI assumptions

Recommended calibration checks:

- Plot predicted uplift decile against observed treatment-control lift.
- Check whether higher predicted uplift groups actually have higher realized lift.
- Compare model rankings using Qini/AUUC rather than raw score magnitude.
- Consider isotonic or regression calibration of uplift scores only after preserving a clean holdout set.

Important interpretation:

> The S-learner, T-learner, X-learner, R-learner, and transformed outcome model are all trying to estimate the same target: expected uplift or CATE. They should be compared by out-of-sample targeting performance, not by which model produces the largest raw uplift values.

---

## Sensitivity and Robustness Checks

Because this dataset is randomized, the main causal identification concern is less severe than in observational data. Still, robustness checks should be included to make the results more convincing.

Recommended checks:

| Check | Purpose |
|---|---|
| Placebo treatment | Randomly shuffle treatment labels and confirm uplift disappears |
| Random cause / irrelevant feature | Add a random feature and confirm it does not dominate the model |
| Subset stability | Re-run models on random subsets and compare conclusions |
| Random replacement / noise | Perturb features and check whether rankings are stable |
| Selection bias simulation | Artificially bias the sample and test how results change |
| Bootstrap confidence intervals | Quantify uncertainty in ATE, decile lift, AUUC, and ROI |
| Segment-level validation | Check if conclusions hold across key customer groups |

CausalML includes sensitivity-analysis utilities that can support checks such as placebo treatment, random cause, subset data, random replacement, and selection bias tests. These should be used as robustness evidence, not as a replacement for the randomized experimental design.

---

## Interpretability Plan

The final output should not just be a score. It should explain which customers should be targeted and why.

Recommended interpretability methods:

### For Response Models

- Feature importance
- Coefficients for logistic regression
- SHAP values, if allowed
- Calibration plots

### For Meta-Learners

- Feature importance from base learners
- Partial dependence of predicted uplift
- Uplift by segment
- Surrogate decision tree trained on predicted uplift scores
- Top-decile customer profile comparison

### For Uplift Trees

- Leaf-level treatment effects
- Treated/control counts per leaf
- Treated/control outcome rates per leaf
- Business description of each high-uplift segment

### Customer Group Interpretation

| Group | Meaning | Recommended Action |
|---|---|---|
| Persuadables | Respond only if treated | Target them |
| Sure things | Respond with or without treatment | Avoid wasting campaign budget |
| Lost causes | Do not respond either way | Do not target |
| Do-not-disturbs | Respond worse when treated | Avoid targeting |

---

## Expected Outputs

The final project should produce:

- Data dictionary
- Treatment/control balance summary
- Campaign performance summary by treatment group
- Segment-level lift analysis
- Average treatment effect estimates
- Traditional response model results
- S-learner uplift estimates
- T-learner uplift estimates
- X-learner uplift estimates, if included
- R-learner or DR-learner results, if included
- Transformed outcome model results
- Uplift tree or causal tree segmentation
- Uplift-by-decile tables
- Qini curves
- AUUC comparison table
- ROI simulation results
- Final targeting recommendation

---

## Suggested Final Model Comparison Table

| Model | Method Type | Outcome | Uplift Metric | AUUC | Qini | Top-Decile Lift | ROI | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| Response model | Predictive baseline | `visit` | Response ranking | TBD | TBD | TBD | TBD | Targets likely visitors, not necessarily persuadables |
| ATE baseline | Causal baseline | `visit` | Constant uplift | TBD | TBD | TBD | TBD | Same treatment effect for everyone |
| T-learner | Meta-learner | `visit` | CATE/uplift | TBD | TBD | TBD | TBD | Separate treated/control models |
| S-learner | Meta-learner | `visit` | CATE/uplift | TBD | TBD | TBD | TBD | Single model with treatment interactions |
| X-learner | Meta-learner | `visit` | CATE/uplift | TBD | TBD | TBD | TBD | Strong optional comparison |
| R-learner | Meta-learner | `visit` | CATE/uplift | TBD | TBD | TBD | TBD | Advanced optional method |
| Transformed outcome | Pseudo-outcome | `visit` | CATE/uplift | TBD | TBD | TBD | TBD | Different method family |
| Uplift tree | Direct uplift | `visit` | Segment lift | TBD | TBD | TBD | TBD | Most interpretable |

---

## Project Roadmap

- [ ] Set up repository structure
- [ ] Add dataset loading and cleaning functions
- [ ] Complete EDA and treatment group summaries
- [ ] Check treatment/control balance with standardized mean differences
- [ ] Estimate ATE for visit, conversion, and spend
- [ ] Build traditional response model baseline
- [ ] Implement T-learner
- [ ] Implement S-learner with treatment interactions
- [ ] Implement transformed outcome model
- [ ] Implement X-learner using CausalML
- [ ] Add optional R-learner or DR-learner
- [ ] Add uplift tree or causal tree model
- [ ] Add uplift-by-decile evaluation
- [ ] Add Qini and AUUC evaluation
- [ ] Add ROI simulation
- [ ] Add sensitivity and robustness checks
- [ ] Add model interpretation visuals
- [ ] Prepare final presentation figures and business recommendation

---

## Key References

- Gutierrez, P. and Gérardy, J.-Y. (2016). *Causal Inference and Uplift Modeling: A Review of the Literature*.
- Hillstrom, K. (2008). *MineThatData E-Mail Analytics and Data Mining Challenge*.
- Imbens, G. W. and Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). *Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning*.
- Radcliffe, N. J. and Surry, P. D. (2011). *Real-World Uplift Modelling with Significance-Based Uplift Trees*.
- Zhao, Z. and Harinen, T. (2019). *Uplift Modeling for Multiple Treatments with Cost Optimization*.
- CausalML documentation. *Uplift modeling and causal inference methods with machine learning*.

---

## Final Business Goal

The final deliverable should answer:

1. Did the email campaign work on average?
2. Which customer segments responded most strongly?
3. Do uplift models rank customers better than ordinary response models?
4. Which modeling method gives the best out-of-sample uplift performance?
5. Which customers should receive an email under an ROI-based targeting rule?
6. Should the business send everyone an email, send no one an email, target by response probability, or target by predicted uplift?

The strongest final recommendation will connect the technical results to a clear business action: target customers with high positive expected uplift, avoid customers with low or negative expected uplift, and use ROI assumptions to decide the cutoff for campaign deployment.
