# IML-Final-project
# 기계학습개론 Final Project: 포항 북구 전세 보증금 예측

포항시 북구 주택 거래 데이터를 활용하여 전세 보증금(월세의 전세 환산 포함) 을 예측하고,  
공간변수(공원, 학교, 바다)가 예측 성능 향상에 기여하는지 분석한 프로젝트입니다.

---

## 1. Project Overview

본 프로젝트는 기계학습개론 Final Project로 수행되었으며,  
주택 기본 특성, 계약 시점, 지역 정보, 그리고 공간 변수를 활용해 전세 보증금을 예측하는 것을 목표로 합니다.

특히 다음 연구 질문을 중심으로 실험을 설계했습니다:

> **공간변수(공원, 학교, 바다)를 추가하면 전세 보증금 예측 성능이 향상되는가?**

---

## 2. Dataset

- **지역:** 경상북도 포항시 북구
- **데이터 구성:** 전세 + 월세 거래
- **타깃 변수:** 전세보증금(월세는 전세환산 방식 적용)
- **시트명:** `최종데이터셋_모델용`

### 전세 환산 방식
월세 거래는 아래 방식으로 전세 보증금으로 환산했습니다.

전세환산보증금 = 월세보증금 + (월세 × 12 / 전환율)

- 사용 전환율: **0.065**

---

## 3. Features

### Group A: Baseline Features
- 전용면적(㎡)
- 층
- 건물연령(계약기준)
- 계약연도
- 계약월
- 바다여부
- 읍면동

### Group B: Spatial Features Added
Group A에 아래 공간변수를 추가했습니다.

- 공원수
- 최대공원면적(㎡)
- 총공원면적(㎡)
- 공원여부
- 동_초등학교수
- 동_중학교수
- 동_고등학교수

---

## 4. Models

다음 3개 모델을 비교했습니다.

- **Ridge Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

---

## 5. Experiment Design

- **Train/Test Split:** 80/20
- **Validation:** 5-Fold Cross Validation
- **Hyperparameter Tuning:** GridSearchCV
- **Evaluation Metrics:**
  - RMSE
  - MAE
  - R²

---

## 6. Main Results

### Performance Summary

| Model | RMSE_A | RMSE_B | ΔRMSE | R²_A | R²_B | ΔR² |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 2974.21 | 2976.01 | +1.80 | 0.8145 | 0.8143 | -0.0002 |
| RandomForest | 2346.10 | 2334.24 | -11.86 | 0.8846 | 0.8857 | +0.0012 |
| XGBoost | 2232.72 | 2245.89 | +13.17 | 0.8955 | 0.8942 | -0.0012 |

### Interpretation
- **XGBoost**가 전체적으로 가장 높은 예측 성능을 보였습니다.
- **공간변수 추가(Group B)** 는 **Random Forest**에서만 소폭 성능 개선을 보였습니다.
- 전체적으로는 **읍면동 변수 자체가 지역 정보를 상당 부분 설명**하고 있어, 공간변수의 추가 효과가 제한적으로 나타난 것으로 해석할 수 있습니다.

---

## 7. Outputs

### Target Distribution
![Target Distribution](outputs/eda_target_dist.png)

### Correlation Heatmap
![Correlation Heatmap](outputs/eda_corr.png)

### Prediction vs Actual
![Prediction vs Actual](outputs/pred_vs_actual.png)

### XGBoost Feature Importance
![XGBoost Feature Importance](outputs/xgb_top_features.png)

---

## 8. Project Structure

```text
ml-intro-jeonse-project/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  └─ jeonse_prediction_pipeline.py
├─ data/
│  └─ IML_Final_dataset.xlsx
├─ outputs/
│  ├─ summary_metrics.csv
│  ├─ results.json
│  ├─ eda_target_dist.png
│  ├─ eda_corr.png
│  ├─ pred_vs_actual.png
│  ├─ xgb_feature_importance.csv
│  └─ xgb_top_features.png
└─ docs/
   ├─ RESULTS_FEEDBACK.md
   ├─ draft_paper.docx
   ├─ dataset_construction_status.docx
   ├─ midterm_presentation.pdf
   └─ course_project_guide.pdf
