# Methodology Revision Notes

본 문서는 `IML_Final_Project_Jeonse_Spatial_Colab.ipynb` 제작 과정에서 기존 방법론 텍스트에 반영해야 하는 논리 수정 사항을 정리한 것이다.

## 1. Target 정의 수정

기존 텍스트는 전세 거래를 primary target으로 설명하지만, 실제 모델링에서는 월세 거래를 전세환산보증금으로 변환해 함께 사용할 수 있다. 따라서 전세-only 분석과 월세 환산 포함 분석을 구분해야 한다.

권장 문장:

> Jeonse transactions are used as the primary target. Monthly rent transactions are converted into Jeonse-equivalent deposits only for the expanded-sample analysis.

전월세 전환율 표기도 명확히 해야 한다. 공식 설명에서는 `%`를 사용하지만, 코드에서는 소수형을 사용한다.

권장 문장:

> In the formula, a 6.5% conversion rate is implemented as `r = 0.065`.

## 2. Group A / Group B 정의 수정

기존 방법론의 핵심 비교는 기본 주택/거래 변수만 포함한 Group A와, 공간 파생변수를 추가한 Group B의 비교다. `읍면동`을 기본 Group A에 포함하면 행정동 위치 정보가 공간 효과를 먼저 흡수할 수 있어, 공간변수의 추가 효과 검정이 흐려진다.

권장 정의:

> Group A includes basic housing and transaction variables such as exclusive area, floor, building age, contract year, and contract month.

> Group B includes all Group A variables plus spatial-derived variables such as sea proximity, park variables, and school variables.

`읍면동`은 메인 Group A/B 비교가 아니라 supplementary location-control analysis로 분리한다.

권장 문장:

> Administrative dong is used only in a supplementary location-control analysis.

## 3. Sea 변수 설명 수정

기존 설명에는 행정동 기반 분류와 아파트 기준 500m 직선거리 기준이 함께 쓰여 있다. 노트북은 이미 모델링 시트에 포함된 `바다여부` 변수를 사용하며, GIS 거리 계산을 새로 수행하지 않는다.

권장 문장:

> Sea proximity is used as an already-derived dummy variable in the modeling dataset. It represents whether the transaction area is classified as coastal or within the predefined 500m coastal proximity rule.

추가 주의:

> This notebook does not recompute GIS distances or verify the 500m threshold from raw coordinates.

## 4. School 변수 수정

기존 데이터 구축 설명에는 school-count range dummy variables가 포함되어 있지만, 현재 모델링 시트에는 해당 구간 더미 기준이 명확히 제공되어 있지 않다. 따라서 노트북에서는 새 구간 더미를 만들지 않는다.

권장 문장:

> This study uses elementary, middle, and high school counts and the dummy indicating whether all three school levels are present. School-count range dummies are not additionally generated because threshold rules are not defined in the provided modeling dataset.

또한 `동_총학교수`는 `동_초등학교수`, `동_중학교수`, `동_고등학교수`의 합이므로 개별 학교수와 동시에 기본 모델에 넣으면 기계적으로 중복된다.

권장 문장:

> Total school count is excluded from the default model when elementary, middle, and high school counts are used separately.

## 5. VIF 설명 보강

VIF를 계산한다고만 쓰면 중복 변수를 어떻게 처리하는지 불명확하다. 방법론에는 중복 변수 처리 기준을 함께 적어야 한다.

권장 문장:

> If a variable is mechanically redundant, such as total school count being the sum of school-level counts, it is excluded from the default feature set.

공원 면적 변수도 VIF 기준으로 정리해야 한다. `총공원면적(㎡)`과 `최대공원면적(㎡)`은 서로 강하게 겹치므로 기본 Group B에 동시에 넣지 않는다. 현재 노트북의 기본 설계에서는 `최대공원면적(㎡)`을 유지하고 `총공원면적(㎡)`은 제외한다.

권장 문장:

> Because total park area and maximum park area show high multicollinearity, the default feature set keeps maximum park area and excludes total park area after VIF screening.

## 6. Scaling 방법 수정

기존 설명은 `StandardScaler`를 사용한다고 되어 있었지만, 이상치에 더 둔감한 처리를 위해 노트북에서는 `RobustScaler`를 사용한다. `RobustScaler`는 중앙값과 IQR 기반으로 스케일링하므로 전세환산보증금 및 면적 변수처럼 분포가 치우친 데이터에서 더 안정적인 선택이다.

권장 문장:

> Numerical variables are scaled using RobustScaler to reduce sensitivity to outliers.

## 7. GridSearchCV 반영

README와 방법론에서 5-fold CV 및 GridSearchCV를 설명한다면, 결과도 고정 파라미터 실행이 아니라 GridSearchCV 기반 결과로 생성해야 한다. 현재 노트북은 Ridge, Random Forest, XGBoost 모두에 대해 `GridSearchCV`와 5-fold cross-validation을 적용한다.

권장 문장:

> Hyperparameters are tuned using GridSearchCV with 5-fold cross-validation on the training set. The selected best estimator is then evaluated on the held-out test set.

## 8. Feature Importance / SHAP 해석 수정

Feature importance와 SHAP는 변수의 예측 기여도를 설명하지만, 인과효과를 증명하지 않는다. 따라서 "영향을 준다"보다 "모델 기반 연관성"으로 해석해야 한다.

권장 문장:

> Feature importance and SHAP are interpreted as model-based associations, not causal effects.

## 9. 결과 해석 기준 수정

Group B 성능이 좋아지면 공간 파생변수의 추가 예측력을 지지할 수 있다. 반대로 성능 개선이 작거나 모델별로 일관되지 않더라도 "공간변수가 의미 없다"고 결론 내리면 안 된다.

권장 문장:

> If the improvement from Group B is small or inconsistent, the study concludes that the additional predictive power of spatial-derived variables is limited under this dataset and model setup, rather than claiming that spatial variables have no relationship with Jeonse prices.

## 10. 노트북에 추가된 Supplementary Analysis

아래 항목은 기존 최소 방법론에는 없지만, 논리적 빈틈을 줄이기 위해 노트북에 별도 supplementary check로 추가했다.

- `읍면동` 포함 location-control comparison
- log target robustness check
- `전월세구분` feature 추가 robustness check
- Jeonse-only robustness check
- VIF 기반 중복 변수 점검
- optional SHAP analysis

이 분석들은 원래 연구 질문을 대체하지 않는다. 메인 결론은 Group A와 Group B의 성능 비교를 기준으로 작성하고, supplementary analysis는 결론의 견고성을 확인하는 용도로만 사용한다.
