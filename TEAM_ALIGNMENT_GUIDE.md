# Team Alignment Guide

이 문서는 보고서, PPT, 코랩, 팀원 개인 분석 자료의 표현을 통일하기 위한 기준입니다.

## 1. 가장 중요한 통일 기준

팀 전체가 아래 정의를 동일하게 사용해야 합니다.

| 항목 | 최종 통일 기준 |
| --- | --- |
| 최종 모델 | XGBoost Regressor |
| 최종 타겟 | `전세환산보증금(만원)` |
| 월세 환산식 | 보증금 + 월세 x 12 / 0.065 |
| 최종 baseline | `A_location_baseline` |
| 최종 spatial model | `B_location_spatial` |
| baseline에 읍면동 포함 여부 | 포함 |
| scaler | 미적용 |
| 튜닝 | GridSearchCV |
| 교차검증 | 5-fold CV |
| 평가 지표 | RMSE, MAE, MAPE, R2 |
| 해석 방법 | SHAP |

## 2. 최종 실험 세팅

### A_location_baseline

기본 비교 기준입니다.

포함 변수:

- 전용면적
- 층
- 건물연령
- 계약연도
- 계약월
- 읍면동

의미:

> 아파트 구조, 계약 시점, 행정동 위치를 이미 통제한 baseline.

### B_location_spatial

최종 공간파생변수 모델입니다.

포함 변수:

- A_location_baseline의 모든 변수
- 바다여부
- 공원여부
- 공원수
- 최대공원면적
- 동_초등학교수
- 동_중학교수
- 동_고등학교수
- 동_초중고모두있음여부

의미:

> 읍면동 위치를 통제한 뒤에도 바다, 공원, 학교 관련 공간파생변수가 추가 예측 정보를 제공하는지 검증하는 모델.

## 3. 팀원 자료에서 그대로 반영해도 되는 것

아래는 최종 설계와 충돌하지 않으므로 PPT/보고서에 반영 가능합니다.

| 팀원 자료 | 반영 방식 |
| --- | --- |
| 전세환산가격 분포 | Data overview 또는 Target distribution 슬라이드 |
| 전세/월세 표본 구성 | Dataset construction 슬라이드 |
| 월별 거래 건수 | Data overview 보조 그래프 |
| 읍면동별 평균 전세값 | Dong-level price differences 슬라이드 |
| 전용면적 vs 전세값 scatter | Housing structure importance 설명 |
| 공원/해안 여부별 boxplot | Spatial variable EDA 또는 appendix |
| 주요 변수 상관관계 heatmap | Multicollinearity/EDA appendix |
| holdout 설명 | Methodology 또는 validation 슬라이드 |
| XGBoost 선택 이유 | Model implementation 슬라이드 |

## 4. 팀원 자료에서 수정해서 써야 하는 것

### 4.1 평균/중앙값 수치

팀원 자료 일부에는 평균 1.64억, 중앙값 1.67억으로 보이는 값이 있습니다.

현재 최종 코랩/PPT asset 기준:

| 항목 | 최종 값 |
| --- | ---: |
| Target mean | 15,707.25만원 |
| Target median | 15,846.15만원 |

보고서와 PPT에는 반드시 최종 코랩 기준 수치를 사용해야 합니다.

만약 팀원 그래프를 그대로 쓴다면, 그래프 안의 평균/중앙값 라벨도 최종 코랩 기준으로 다시 생성해야 합니다.

### 4.2 `B_spatial_no_dong` 표현

팀원 자료의 `B_spatial_no_dong`은 최종 메인 모델이 아닙니다.

최종 메인 비교:

> `B_location_spatial - A_location_baseline`

따라서 발표와 보고서 본문에서는 `B_spatial_no_dong`을 메인 결과처럼 쓰지 않습니다.

가능한 사용 방식:

> Earlier exploratory comparison or appendix only.

## 5. 팀원 자료에서 메인에 넣으면 안 되는 것

### Surrogate tree

팀원 자료의 surrogate tree는 다음 이유로 메인 결과에 넣으면 안 됩니다.

1. 최종 모델인 XGBoost 자체가 아니라 XGBoost를 단순화해서 흉내 낸 대리모델입니다.
2. 일부 그림은 `B_spatial_no_dong` 기준이라 최종 설계인 `B_location_spatial`과 다릅니다.
3. 메인 해석은 이미 SHAP으로 수행하고 있으므로, surrogate tree를 앞세우면 해석 기준이 흔들립니다.

사용한다면 appendix에서만 아래처럼 설명합니다.

> As a supplementary interpretability check, a shallow surrogate tree was fitted to approximate the XGBoost prediction pattern. This is not the final model itself and should not be interpreted as the main evidence for spatial-variable effects.

## 6. PPT 담당자용 통일 문구

### Feature set slide

Use:

> Dong is included in the baseline as a location control. The spatial model tests whether sea, park, and school-related variables provide additional information beyond dong identity.

Do not use:

> Baseline does not include location variables.

### Main result slide

Use:

> The spatial model improves every metric, but the random-split gain is small.

Do not use:

> Spatial variables dramatically improve prediction in the random split.

### Holdout slide

Use:

> Spatial variables are more useful for generalization than for producing a large random-split gain.

Do not use:

> We selected the holdout setting with the best result.

### SHAP slide

Use:

> Housing structure is the dominant predictor, while school and park variables add supplementary local-context information.

Do not use:

> Spatial variables are the most important predictors.

## 7. 보고서 담당자용 통일 문구

Use this paragraph in the methodology section:

> This study compares two XGBoost feature settings. The baseline model, A_location_baseline, includes apartment structural variables, contract timing variables, and administrative dong. The spatial model, B_location_spatial, adds sea, park, and school-related spatial-derived variables. This design evaluates whether spatial-derived variables provide additional predictive information after controlling for administrative dong-level location.

Use this paragraph in the results section:

> The spatial model slightly improves all metrics in the random split, but the improvement is small because dong-level location is already controlled. However, the spatial model shows stronger improvements under stricter holdout validation, including complex, dong, and time holdout settings. Therefore, the results suggest that spatial-derived variables are more useful for generalization than for producing a large random-split gain.

Use this paragraph in the interpretation section:

> SHAP analysis indicates that housing structure is the dominant source of prediction. Among spatial-derived variables, school and park features provide additional local-context information, while sea proximity has limited contribution in this dataset. These results should be interpreted as predictive evidence rather than causal evidence.

## 8. 팀원에게 전달할 짧은 버전

팀원에게는 아래 문장을 그대로 보내면 됩니다.

> 최종본은 `A_location_baseline`과 `B_location_spatial` 비교로 통일하겠습니다. 즉 baseline에도 읍면동을 포함하고, 공간파생변수는 읍면동을 통제한 뒤 추가되는 변수로 설명합니다. 팀원 EDA 그래프는 데이터 설명과 appendix에 반영하되, `B_spatial_no_dong surrogate tree`는 최종 메인 결과가 아니므로 본문 결과로 쓰지 않고 보조 해석 또는 부록으로만 사용하겠습니다. 수치는 최종 코랩 기준으로 통일해야 하므로 평균/중앙값이 다른 그래프는 다시 생성하거나 라벨을 수정해야 합니다.

## 9. 최종 원칙

최종 발표와 보고서는 아래 한 문장으로 정리됩니다.

> 공간파생변수는 읍면동 위치 통제를 대체하지 않지만, 학교와 공원 중심의 지역 맥락 정보를 추가로 제공하며, 특히 더 엄격한 holdout 검증에서 일반화 성능 개선에 기여한다.
