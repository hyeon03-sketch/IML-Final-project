# Final Handoff for Report and Presentation

Project title:
Verification of Additional Predictive Power of Spatial-Derived Variables:
A Study on Jeonse Price Prediction in Buk-gu, Pohang

## 1. Final Research Question

Use this research question consistently in the report and PPT:

> Do spatial-derived variables provide additional predictive power for jeonse price prediction after controlling for apartment characteristics, contract timing, and administrative dong-level location?

Korean explanation:

> 본 연구는 포항시 북구 아파트 전세환산보증금 예측에서, 전용면적·층·건물연령·계약시점·읍면동 정보를 통제한 뒤에도 바다, 공원, 학교 관련 공간파생변수가 추가적인 예측 정보를 제공하는지를 검증한다.

## 2. Final Dataset and Target

Use the final dataset statistics generated from the final Colab/PPT asset pipeline.

| Item | Final value |
| --- | ---: |
| Total observations | 4,470 |
| Jeonse transactions | 2,486 |
| Monthly-rent transactions converted | 1,984 |
| Number of dongs | 26 |
| Target mean | 15,707.25 |
| Target median | 15,846.15 |

Target variable:

> `전세환산보증금(만원)` / jeonse-equivalent deposit in 10,000 KRW.

Target construction:

| Transaction type | Target construction |
| --- | --- |
| Jeonse | jeonse-equivalent deposit = deposit |
| Monthly rent | jeonse-equivalent deposit = deposit + monthly rent x 12 / 0.065 |

Important note:

> If teammate visualizations show mean 1.64억 or median 1.67억, do not mix those values with the final report unless the same final Colab is rerun and produces those values. The final PPT/report should use one consistent set of numbers.

## 3. Final Feature Set Design

The final comparison is not a simple "basic variables vs spatial variables" comparison.
It is a conservative test where dong-level location is already included in the baseline.

| Experiment | Purpose | Included variables |
| --- | --- | --- |
| `A_location_baseline` | Location-controlled baseline | exclusive area, floor, building age, contract year, contract month, administrative dong |
| `B_location_spatial` | Spatial-extended model | A + sea dummy, park dummy, park count, max park area, elementary/middle/high school counts, all-school-levels dummy |

Use this wording:

> Since the study focuses on neighborhood-level differences within Buk-gu, administrative dong is included in the baseline model as a location control. Spatial-derived variables are then added to test whether they provide additional information beyond dong identity.

Do not describe Group A as excluding location. That was an earlier version and should not be used as the final methodology.

## 4. Final Model and Implementation

Final model family:

> XGBoost Regressor.

Implementation settings:

| Item | Final setting |
| --- | --- |
| Model | XGBoost Regressor |
| Train/test split | 80/20 random split for main result |
| Hyperparameter tuning | GridSearchCV |
| Cross-validation | 5-fold CV |
| Metrics | RMSE, MAE, MAPE, R2 |
| Scaler | Not applied |
| Categorical handling | One-hot encoding for `읍면동` |
| Binary variables | Passed as 0/1 |

Why no scaler:

> Feature scaling was not applied because the final model is tree-based XGBoost. Tree splits are based on threshold ordering rather than distance or gradient magnitude across differently scaled variables.

Why XGBoost:

> XGBoost was selected because it can capture nonlinear relationships and interaction effects in tabular housing data, while also supporting SHAP-based interpretation.

## 5. Final Main Result

Use these final main-result values.

| Experiment | RMSE | MAE | MAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| A_location_baseline | 2,216.95 | 1,518.12 | 11.40% | 0.8969 |
| B_location_spatial | 2,208.91 | 1,515.95 | 11.39% | 0.8977 |

Main change:

| Metric | B - A |
| --- | ---: |
| RMSE | -8.04 |
| MAE | -2.17 |
| MAPE | -0.014%p |
| R2 | +0.0007 |

Correct interpretation:

> In the random split, the improvement from spatial-derived variables is positive but small. This is expected because administrative dong already captures broad neighborhood-level location information. Therefore, the main claim should not overstate the random-split gain.

## 6. Final Holdout Validation

Use stricter holdout tests to show whether the spatial model helps generalization.

| Validation | RMSE change | MAE change | MAPE change | R2 change |
| --- | ---: | ---: | ---: | ---: |
| complex holdout | -288.88 | -163.62 | -0.754%p | +0.0505 |
| dong holdout | -138.62 | -255.44 | -1.996%p | +0.0250 |
| random split | -8.04 | -2.17 | -0.014%p | +0.0007 |
| time holdout 2026 | -94.57 | -64.76 | -0.251%p | +0.0098 |

Correct interpretation:

> The spatial model improves all stricter validation settings. The improvement is larger under complex, dong, and time holdout validation than under the simple random split. This suggests that spatial-derived variables are more useful for generalization than for producing a large random-split gain.

Do not write:

> "We selected the holdout with the best result."

Use instead:

> "We conducted multiple holdout validations to check whether the result was robust under stricter train-test separation."

## 7. Final SHAP Interpretation

Use SHAP as the main interpretability result.

| Feature group | SHAP share |
| --- | ---: |
| Housing structure | 75.63% |
| School | 11.73% |
| Dong location | 5.35% |
| Park | 5.02% |
| Contract time | 1.86% |
| Sea | 0.41% |

Correct interpretation:

> Housing structure is the dominant source of prediction. However, school and park variables contribute nontrivial local-context information. Sea proximity contributes little in this dataset.

Use this careful conclusion:

> Spatial-derived variables do not replace housing-structure or dong-location variables. Instead, they provide supplementary local-context information, especially through school and park-related features.

## 8. What to Do With Teammate Materials

### Reflect in main PPT/report

Use these teammate materials because they support the final story:

1. Jeonse-equivalent price distribution
2. Jeonse/monthly-rent sample composition
3. Monthly transaction count, if space allows
4. Dong-level average jeonse-equivalent deposit
5. Exclusive area vs jeonse-equivalent deposit scatter
6. Park/sea dummy boxplots
7. Correlation heatmap
8. Holdout validation explanations
9. XGBoost explanation based on nonlinear relationships

### Use only in appendix

Surrogate tree:

> Use only as supplementary interpretability material. It is not the final XGBoost model and should not be used as the main result.

Required wording if included:

> As a supplementary interpretability check, a shallow surrogate tree was fitted to approximate the XGBoost prediction pattern. The tree suggests that building age and exclusive area dominate the simplified decision structure. However, this is not the final model itself and should not be interpreted as the main evidence for spatial-variable effects.

### Do not include as main evidence

Do not use `B_spatial_no_dong surrogate tree` as the main interpretation figure.

Reason:

> The final methodology controls for administrative dong in the baseline. A no-dong surrogate tree uses a different feature setting and can conflict with the final experimental design.

## 9. Recommended PPT Structure After Slide 12

Use this structure for the final presentation.

13. Final Dataset
    - Dataset size, jeonse/monthly-rent converted rows, number of dongs
    - Include transaction composition chart

14. Target Construction
    - Explain jeonse-equivalent deposit formula
    - Include target distribution chart

15. Spatial-Derived Variables
    - Sea, park, school, education mix variables

16. Dong-Level Consistency Check
    - Explain harmonization of dong-level variables

17. Feature Set Design
    - A_location_baseline vs B_location_spatial
    - Emphasize dong as baseline location control

18. Dong-Level Price Differences
    - Use dong average target chart
    - Message: jeonse price is spatially patterned

19. XGBoost Modeling Pipeline
    - Dataset -> feature sets -> XGBoost -> GridSearchCV -> metrics -> SHAP

20. Metrics and Tuning
    - RMSE, MAE, MAPE, R2
    - No scaler, one-hot dong, GridSearchCV

21. Main XGBoost Result
    - Main result table and metrics chart
    - Message: improvement is positive but small in random split

22. How to Read the Main Result
    - Dong already captures broad location information
    - Spatial variables add supplementary information

23. Additional Holdout Validation
    - Explain random/time/complex/dong holdout

24. Holdout Validation Results
    - Use holdout delta table/bar chart
    - Main evidence: stricter validation improves

25. Prediction vs Actual
    - Show A vs B prediction plots

26. Why SHAP Was Used
    - Explain model interpretability

27. SHAP Feature Group Results
    - Use SHAP group table/bar chart

28. Spatial Variables in SHAP
    - Use top SHAP features
    - Message: school and park matter more than sea

29. Main Findings
    - Strong XGBoost performance
    - Modest random-split spatial gain
    - Stronger holdout evidence
    - Housing structure dominates
    - School/park provide local context

30. Limitations and Conclusion
    - Fixed conversion rate
    - Dong-level spatial variables
    - Sea dummy limitations
    - SHAP is not causality

Appendix:
    - Correlation heatmap
    - Exclusive area scatter
    - Park/sea boxplots
    - Surrogate tree

## 10. Report Structure and Copy-Ready Text

### Introduction

> This study examines whether spatial-derived variables improve jeonse price prediction in Buk-gu, Pohang. While apartment-level structural characteristics such as exclusive area, floor, and building age are important predictors, local neighborhood conditions may also affect jeonse price formation. Therefore, this study tests whether sea proximity, park-related variables, and school-related variables provide additional predictive power beyond apartment attributes, contract timing, and administrative dong-level location.

### Data and Target

> The modeling dataset consists of 4,470 apartment lease transactions in Buk-gu, Pohang, including 2,486 jeonse transactions and 1,984 monthly-rent transactions converted into jeonse-equivalent deposits. The target variable is jeonse-equivalent deposit measured in 10,000 KRW. For jeonse transactions, the target equals the deposit. For monthly-rent transactions, the target is calculated as deposit plus monthly rent multiplied by 12 and divided by the fixed conversion rate of 0.065.

### Methodology

> Two XGBoost feature settings were compared. The baseline model, A_location_baseline, includes apartment structure variables, contract timing variables, and administrative dong. The spatial model, B_location_spatial, adds sea, park, and school-related spatial-derived variables. This design tests whether spatial-derived variables provide additional information after controlling for dong-level location.

### Validation

> In addition to the main 80/20 random split, stricter holdout validations were conducted. Time holdout evaluates prediction for the latest contract year, complex holdout prevents the same apartment complex from appearing in both train and test sets, and dong holdout prevents the same administrative dong from appearing in both train and test sets. These validations reduce the risk that the model is simply memorizing repeated local patterns.

### Results

> In the random split, the spatial model slightly improved all evaluation metrics. RMSE decreased from 2,216.95 to 2,208.91, MAE decreased from 1,518.12 to 1,515.95, MAPE decreased from 11.40% to 11.39%, and R2 increased from 0.8969 to 0.8977. Although the random-split improvement is small, the spatial model showed stronger improvements under stricter holdout validation.

### Interpretation

> SHAP analysis shows that housing structure is the dominant source of prediction, accounting for 75.63% of the grouped SHAP contribution. Among spatial-related variables, school variables account for 11.73% and park variables account for 5.02%. Sea proximity contributes only 0.41%. These results suggest that spatial-derived variables are not the main predictors, but school and park variables provide meaningful supplementary local-context information.

### Conclusion

> The results indicate that spatial-derived variables provide modest but consistent additional predictive power for jeonse price prediction. Their value is clearest under stricter validation settings, where local-context variables help generalization beyond simple random split performance. However, the findings should be interpreted as predictive evidence, not causal evidence.

## 11. Final Do-Not-Mix Rules

1. Do not mix final dataset statistics with teammate statistics if means/medians differ.
2. Do not present no-dong surrogate tree as the final model.
3. Do not claim spatial variables are the dominant predictors.
4. Do not claim causal effects from SHAP or boxplots.
5. Do not say the model selected the best holdout result.
6. Do not return to Ridge/RandomForest as final models unless placed in an appendix as earlier model screening.

## 12. Final One-Sentence Thesis

Use this as the final presentation message:

> Spatial-derived variables do not replace housing attributes or dong-level location controls, but they add supplementary local-context information that improves XGBoost jeonse price prediction, especially under stricter holdout validation.
