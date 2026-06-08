# Dong-Included Baseline Report Section Guide

Colab 실행 링크:

https://colab.research.google.com/github/hyeon03-sketch/IML-Final-project/blob/codex/ml-project-review/notebooks/IML_Final_Project_Dong_Baseline_XGBoost_SHAP_Colab.ipynb

## 1. 연구 목적 / Introduction

넣을 위치:

- 연구 배경 뒤
- Research Question 앞 또는 Research Question 안

삽입 문구:

> 본 연구는 포항시 북구의 전세가격을 읍면동 단위의 지역적 특성과 함께 분석하는 것을 목적으로 한다. 따라서 전세가격 예측에서 `읍면동`은 단순 보조변수가 아니라 지역 차이를 반영하는 핵심 독립변수로 간주한다. 본 연구는 기본 주택 특성과 읍면동 정보를 포함한 baseline 모델에 바다, 공원, 학교 등 공간파생변수를 추가했을 때 예측 성능이 개선되는지를 검증한다.

Research Question 문구:

> RQ. 읍면동 정보를 포함한 baseline 모델에 공간파생변수를 추가하면 포항시 북구 전세환산보증금 예측 성능이 개선되는가?

## 2. 데이터 구축 / Dataset

넣을 위치:

- Housing transaction data 설명 뒤
- Target variable 설명 부분

삽입 문구:

> 본 연구의 종속변수는 `전세환산보증금(만원)`이다. 전세 거래는 기존 보증금을 그대로 사용하고, 월세 거래는 전월세전환율 6.5%를 적용하여 전세환산보증금으로 변환하였다. 이를 통해 전세와 월세 거래를 하나의 연속적인 주거비 부담 지표로 통합하였다.

수식:

```text
전세환산보증금 = 보증금(만원) + 월세금(만원) * 12 / 0.065
```

데이터 정합성 문구:

> 공간파생변수는 읍면동 단위 변수이므로 같은 읍면동 안에서는 값이 일정해야 한다. 데이터 점검 결과 일부 동에서 학교 수 값이 일관되지 않은 행이 발견되어, 해당 변수는 읍면동별 최빈값으로 통일하였다.

## 3. EDA / 지역별 분석

넣을 위치:

- 기존 EDA 섹션
- 모델링 전에 새 소제목으로 추가

추천 소제목:

```text
Dong-level Exploratory Analysis
```

삽입 문구:

> 본 연구는 기간별 예측보다 지역별 전세가격 차이에 초점을 두기 때문에, 읍면동 단위의 탐색적 분석을 추가하였다. 각 읍면동별 평균 전세환산보증금, 거래 수, 평균 전용면적, 평균 건물연령, 공원·학교·바다 관련 변수를 집계하고, 동 단위 상관관계를 확인하였다. 이 분석은 인과관계를 검정하기 위한 것이 아니라, 지역별 가격 차이와 공간환경 변수 사이의 관계를 기술적으로 보여주기 위한 것이다.

넣을 표/그림:

| 자료 | Colab 출력 |
|---|---|
| 읍면동별 평균 전세환산보증금 막대그래프 | `dong_mean_target_bar.png` |
| 읍면동 단위 상관관계 heatmap | `dong_level_correlation_heatmap.png` |
| 읍면동 집계표 | `dong_summary.csv` |

주의 문구:

> 상관관계 분석은 읍면동 단위의 기술적 분석이며, 개별 거래 수준의 인과효과를 의미하지 않는다.

## 4. Methodology / Feature Set Design

넣을 위치:

- 기존 Group A/B 설명 부분 교체

삽입 문구:

> 회의 피드백을 반영하여, 본 연구의 baseline에는 `읍면동` 정보를 포함하였다. 이는 본 연구가 포항시 북구 내 지역별 전세가격 차이를 분석하는 것이므로, 행정동 정보가 기본적인 지역 특성을 반영하는 핵심 변수이기 때문이다. 따라서 최종 실험은 `읍면동`을 제외한 baseline이 아니라, `읍면동`을 포함한 baseline에 공간파생변수를 추가하는 방식으로 설계하였다.

Feature set 표:

| Group | 포함 변수 | 역할 |
|---|---|---|
| `A_location_baseline` | 전용면적, 층, 건물연령, 계약연도, 계약월, 읍면동 | 기본 지역 통제 baseline |
| `B_location_spatial` | A + 바다여부, 공원여부, 공원수, 최대공원면적, 초/중/고 학교수, 초중고모두있음여부 | 공간파생변수 추가 모델 |

해석 문구:

> 이 비교는 행정동 정보만으로 설명되지 않는 추가적인 지역 환경 정보가 예측 성능을 보완하는지 확인하는 보수적 검증이다. 공간파생변수 대부분이 동 단위 변수이기 때문에, `읍면동`을 포함한 뒤의 성능 개선 폭은 작을 수 있다.

## 5. Model Implementation

넣을 위치:

- 모델 구현 섹션

삽입 문구:

> 최종 모델은 XGBoost Regressor로 고정하였다. 이전 실험에서 XGBoost가 가장 높은 예측 성능을 보였기 때문에, 본 연구에서는 여러 모델을 비교하기보다 XGBoost 안에서 feature set 차이에 따른 성능 변화를 비교하였다.

설명할 구현 요소:

| 항목 | 내용 |
|---|---|
| 모델 | XGBoost Regressor |
| Split | 80/20 random train-test split |
| Tuning | 5-fold CV 기반 GridSearchCV |
| 평가 지표 | RMSE, MAE, MAPE, R² |
| Categorical 처리 | `읍면동` One-Hot Encoding |
| Scaler | XGBoost는 tree-based model이므로 미적용 |

## 6. Results

넣을 위치:

- 모델 성능 결과 섹션

삽입 문구 템플릿:

> `A_location_baseline`과 `B_location_spatial`을 비교한 결과, 공간파생변수를 추가한 모델에서 RMSE, MAE, MAPE가 감소하고 R²가 증가하였다. 개선 폭은 크지 않지만, 이는 `읍면동` 정보가 이미 지역적 차이를 상당 부분 설명하기 때문으로 해석할 수 있다. 따라서 공간파생변수는 행정동 정보와 중복되는 일부 정보를 가지면서도, 추가적인 지역 환경 정보를 제공하는 보완 변수로 볼 수 있다.

넣을 표:

| Model setting | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| `A_location_baseline` | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 |
| `B_location_spatial` | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 |

Delta 표:

| Comparison | ΔRMSE | ΔMAE | ΔMAPE | ΔR² |
|---|---:|---:|---:|---:|
| `B_location_spatial - A_location_baseline` | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 |

## 7. SHAP Interpretation

넣을 위치:

- Feature importance / interpretation 섹션

삽입 문구:

> SHAP 분석은 `B_location_spatial` 모델을 기준으로 수행하였다. 이는 해당 모델이 읍면동과 공간파생변수를 모두 포함하므로, 행정동 위치정보와 공간환경 변수의 상대적 기여도를 함께 확인할 수 있기 때문이다.

해석 문구 템플릿:

> SHAP 결과에서 가장 큰 기여도를 보인 변수군은 주택 구조 변수였다. 이는 전세가격 예측에서 건물연령, 전용면적, 층과 같은 아파트 자체 특성이 핵심적임을 의미한다. 공간파생변수는 주된 예측 요인이라기보다, 지역 환경 정보를 보완하는 변수로 나타났다.

공간변수 비중 문구:

> 학교, 공원, 바다 관련 변수의 SHAP 비중을 합산하여 공간파생변수의 전체 기여도를 확인하였다. 이 값은 공간파생변수가 모델 예측에서 차지하는 상대적 중요도를 보여준다.

## 8. Robustness Check

넣을 위치:

- Results 뒤 또는 Appendix 직전

삽입 문구:

> Random split 결과가 동일 단지 또는 유사 거래의 중복으로 과대평가될 가능성을 점검하기 위해 시간 기준, 단지 기준, 동 기준 holdout 검증을 추가하였다. 이를 통해 공간파생변수 추가 효과가 random split에만 의존하는지 확인하였다.

표:

| Validation | ΔRMSE | ΔMAE | ΔMAPE | ΔR² | 해석 |
|---|---:|---:|---:|---:|---|
| random split | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | 개선 여부 |
| time holdout | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | 개선 여부 |
| complex holdout | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | 개선 여부 |
| dong holdout | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | Colab 결과 입력 | 개선 여부 |

## 9. Conclusion

삽입 문구:

> 본 연구는 포항시 북구의 전세환산보증금 예측에서 읍면동 정보를 포함한 baseline 모델에 공간파생변수를 추가하는 방식으로 지역 환경 정보의 예측 기여도를 검증하였다. 실험 결과, 공간파생변수는 예측 성능을 보완하는 효과를 보였으나, 그 효과는 읍면동 정보와 일부 중복되어 개선 폭이 크지는 않았다. 따라서 공간파생변수는 전세가격 예측의 단독 핵심 요인이라기보다, 행정동 위치정보와 주택 구조 변수 중심 모델을 보완하는 지역 환경 정보로 해석하는 것이 타당하다.

## 10. 발표용 핵심 문장

> 저희 연구는 지역별 전세가격 차이를 보는 연구이기 때문에 `읍면동`을 baseline에 포함했고, 그 위에 공간파생변수를 추가했습니다. 결과적으로 공간파생변수는 읍면동 정보와 일부 겹치기 때문에 개선 폭은 크지 않았지만, XGBoost와 SHAP 분석에서 지역 환경 정보를 보완하는 변수로 확인되었습니다.

