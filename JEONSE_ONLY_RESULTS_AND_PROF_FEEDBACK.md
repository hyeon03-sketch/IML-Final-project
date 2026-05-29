# Jeonse-Only Experiment Results and Professor Feedback Questions

본 문서는 `IML_Final_Project_Jeonse_Only_NoScaler_Colab.ipynb` 실행 결과를 바탕으로, 교수님께 확인받아야 할 사항을 정리한 것이다.

## 1. 현재 실험 설정 요약

이번 실험은 월세 거래를 모두 제외하고, 전세 거래만 사용했다.

- 전체 원자료: 4,470건
- 전세-only 사용 데이터: 2,486건
- 제거된 월세 거래: 1,984건
- 읍면동 수: 22개
- 계약연도: 2025, 2026
- 결측치: 0
- target: `보증금(만원)`
- 월세 -> 전세환산: 사용하지 않음
- scaler: 사용하지 않음
- 모델: Ridge, RandomForest, XGBoost
- 검증: 80/20 random train-test split, 5-fold CV, GridSearchCV

## 2. Feature Set

### Main Comparison

| Group | 포함 변수 |
|---|---|
| `A_baseline` | 전용면적, 층, 건물연령, 계약연도, 계약월 |
| `B_spatial_extended` | A + 바다여부 + 공원여부 + 공원수 + 최대공원면적 + 초/중/고 학교수 + 초중고모두있음여부 |

### Supplementary Location-Control Comparison

| Group | 포함 변수 |
|---|---|
| `A_location_control_with_dong` | A + 읍면동 |
| `B_spatial_location_control_with_dong` | B + 읍면동 |

`읍면동`은 main comparison에는 넣지 않고, supplementary control로만 사용했다. 이유는 `읍면동`이 지역 정보를 강하게 흡수해 공간 파생변수의 추가 효과를 가릴 수 있기 때문이다.

## 3. VIF 처리

VIF 확인 결과, 학교 변수에서 완전 다중공선성이 발생했다.

- `동_총학교수 = 동_초등학교수 + 동_중학교수 + 동_고등학교수`
- 따라서 `동_총학교수`는 기본 모델에서 제외했다.

공원 면적 변수도 VIF가 높았다.

- `총공원면적(㎡)` VIF 약 22.6
- `최대공원면적(㎡)` VIF 약 11.7

따라서 기본 Group B에서는 둘을 동시에 쓰지 않고, `최대공원면적(㎡)`만 유지하고 `총공원면적(㎡)`은 제외했다.

## 4. 주요 성능 결과

### Best Model by Feature Group

| Experiment | Best Model | CV RMSE | Test RMSE | Test MAE | Test MAPE | Test R² |
|---|---|---:|---:|---:|---:|---:|
| A_baseline | XGBoost | 2225.85 | 1900.12 | 1252.99 | 8.27% | 0.9322 |
| A_location_control_with_dong | XGBoost | 2028.60 | 1716.04 | 1147.12 | 7.49% | 0.9447 |
| B_spatial_extended | XGBoost | 2026.75 | 1703.67 | 1122.40 | 7.31% | 0.9455 |
| B_spatial_location_control_with_dong | XGBoost | 2023.35 | 1744.20 | 1139.99 | 7.49% | 0.9429 |

가장 좋은 성능은 `B_spatial_extended`의 XGBoost에서 나왔다.

- Test RMSE: 약 1,704만원
- Test MAE: 약 1,122만원
- Test MAPE: 약 7.31%
- Test R²: 약 0.9455

### Model-wise Full Comparison

| Experiment | Model | CV RMSE | Test RMSE | Test MAE | Test MAPE | Test R² |
|---|---|---:|---:|---:|---:|---:|
| A_baseline | Ridge | 3600.39 | 3547.17 | 2803.39 | 17.88% | 0.7638 |
| A_baseline | RandomForest | 2233.30 | 1900.32 | 1177.28 | 7.55% | 0.9322 |
| A_baseline | XGBoost | 2225.85 | 1900.12 | 1252.99 | 8.27% | 0.9322 |
| B_spatial_extended | Ridge | 2809.36 | 2678.53 | 1989.58 | 12.79% | 0.8653 |
| B_spatial_extended | RandomForest | 2083.53 | 1855.48 | 1076.68 | 6.97% | 0.9354 |
| B_spatial_extended | XGBoost | 2026.75 | 1703.67 | 1122.40 | 7.31% | 0.9455 |
| A_location_control_with_dong | Ridge | 2701.73 | 2565.44 | 1886.65 | 12.76% | 0.8764 |
| A_location_control_with_dong | RandomForest | 2079.04 | 1751.99 | 1038.84 | 6.67% | 0.9424 |
| A_location_control_with_dong | XGBoost | 2028.60 | 1716.04 | 1147.12 | 7.49% | 0.9447 |
| B_spatial_location_control_with_dong | Ridge | 2704.09 | 2561.86 | 1886.32 | 12.76% | 0.8768 |
| B_spatial_location_control_with_dong | RandomForest | 2072.42 | 1814.92 | 1045.24 | 6.79% | 0.9382 |
| B_spatial_location_control_with_dong | XGBoost | 2023.35 | 1744.20 | 1139.99 | 7.49% | 0.9429 |

모델별 해석:

- RMSE와 R² 기준으로는 `B_spatial_extended`의 XGBoost가 가장 좋다.
- MAE와 MAPE 기준으로는 `A_location_control_with_dong`의 RandomForest가 가장 낮지만, 이 실험은 `읍면동`을 통제변수로 넣은 supplementary setting이다.
- 연구 질문의 핵심인 `A_baseline` vs `B_spatial_extended`에서는 Ridge, RandomForest, XGBoost 모두 공간변수 추가 후 성능이 개선됐다.
- 따라서 발표에서는 “최고 성능 모델은 XGBoost + 공간변수”라고 말하고, “오차 크기 관점에서는 RandomForest도 안정적”이라고 보조적으로 설명하는 것이 안전하다.

## 5. Main Group A/B 비교

공간변수 추가 효과를 보는 핵심 비교는 `A_baseline`과 `B_spatial_extended`이다.

| Model | ΔRMSE | ΔMAE | ΔMAPE | ΔR² |
|---|---:|---:|---:|---:|
| RandomForest | -44.84 | -100.60 | -0.58%p | +0.0032 |
| Ridge | -868.64 | -813.81 | -5.09%p | +0.1015 |
| XGBoost | -196.45 | -130.59 | -0.96%p | +0.0133 |

해석:

- 세 모델 모두에서 Group B가 Group A보다 RMSE, MAE, MAPE가 낮아졌다.
- 세 모델 모두에서 Group B의 R²가 높아졌다.
- 따라서 random split 기준에서는 공간 파생변수 추가가 예측 성능 개선에 기여했다고 볼 수 있다.

## 6. Supplementary Location-Control 결과

`읍면동`을 통제한 상태에서는 공간변수 추가 효과가 거의 사라지거나 오히려 약해졌다.

| Model | ΔRMSE | ΔMAE | ΔMAPE | ΔR² |
|---|---:|---:|---:|---:|
| RandomForest | +62.94 | +6.40 | +0.12%p | -0.0042 |
| Ridge | -3.57 | -0.33 | -0.00%p | +0.0003 |
| XGBoost | +28.16 | -7.13 | +0.00%p | -0.0018 |

해석:

- `읍면동`이 들어가면 지역 차이를 이미 많이 설명하기 때문에, 학교/공원/바다 변수의 추가 효과가 약해지는 것으로 보인다.
- 이 결과는 공간변수가 무의미하다는 뜻이 아니라, 행정동 변수와 공간 파생변수가 일부 중복 정보를 가진다는 의미로 해석하는 것이 안전하다.

## 7. 추가 Robustness Validation 필요

현재 성능은 random split 기준이다. 하지만 전세 거래 데이터는 같은 단지, 같은 동, 비슷한 계약 시점의 거래가 반복될 수 있으므로, random split에서는 train/test에 매우 유사한 표본이 함께 들어갈 수 있다. 따라서 학술적으로 강한 결론을 내려면 아래 검증을 추가로 확인해야 한다.

| 검증 방식 | 목적 | 유지되어야 할 결과 |
|---|---|---|
| 시간 기준 holdout | 과거 연도 거래로 학습하고 이후 연도 거래를 예측할 수 있는지 확인 | `B_spatial_extended`가 `A_baseline`보다 RMSE/MAPE가 낮거나 최소한 비슷해야 함 |
| 단지 기준 holdout | 학습에 없던 아파트 단지에도 일반화되는지 확인 | 성능은 떨어질 수 있지만, 공간변수 추가 효과의 방향이 유지되는지 확인 |
| 동 기준 holdout | 학습에 없던 행정동에도 일반화되는지 확인 | 가장 엄격한 검증이므로 성능 하락 가능성이 크며, 결과 해석 시 별도 논의 필요 |

노트북에는 `Required Robustness Validation` 섹션을 추가했고, 기본값을 `RUN_STRICT_VALIDATION = True`로 설정했다. 실행하면 다음 두 파일이 생성된다.

- `strict_validation_results.csv`: 검증 방식별 A/B 모델 성능
- `strict_validation_delta.csv`: 검증 방식별 `B_spatial_extended - A_baseline` 차이

판정 기준:

- random split에서만 좋아지고 holdout에서는 나빠진다면, “공간변수가 예측력을 개선한다”는 주장은 약해진다.
- 시간/단지 holdout에서도 개선 방향이 유지되면, 학술적으로 훨씬 설득력이 커진다.
- 동 holdout에서 성능이 떨어지는 것은 자연스러울 수 있다. 이 경우 “새로운 동으로의 외삽은 어렵지만, 관측된 지역 내에서는 공간변수가 설명력을 보완한다”로 조심스럽게 해석해야 한다.

### Strict Validation 실행 결과

아래 결과는 전세-only, no-scaler, XGBoost 기준으로 `A_baseline`과 `B_spatial_extended`를 다시 비교한 것이다. Random split 결과는 실행 환경과 XGBoost 버전에 따라 위의 Colab 실행 결과와 소폭 차이가 날 수 있으나, 비교 방향은 동일하다.

| Validation | Group | Train N | Test N | RMSE | MAE | MAPE | R² |
|---|---|---:|---:|---:|---:|---:|---:|
| random_split | A_baseline | 1,988 | 498 | 1916.40 | 1336.33 | 8.84% | 0.9311 |
| random_split | B_spatial_extended | 1,988 | 498 | 1720.36 | 1136.52 | 7.38% | 0.9444 |
| time_holdout_test_2026 | A_baseline | 1,772 | 714 | 3403.40 | 2463.50 | 13.76% | 0.7884 |
| time_holdout_test_2026 | B_spatial_extended | 1,772 | 714 | 2535.17 | 1830.21 | 11.12% | 0.8826 |
| complex_holdout | A_baseline | 2,225 | 261 | 3012.68 | 2390.38 | 18.24% | 0.7289 |
| complex_holdout | B_spatial_extended | 2,225 | 261 | 2696.43 | 2189.71 | 17.44% | 0.7828 |
| dong_holdout | A_baseline | 2,102 | 384 | 4966.65 | 3745.07 | 24.07% | 0.3508 |
| dong_holdout | B_spatial_extended | 2,102 | 384 | 3227.06 | 2441.75 | 17.16% | 0.7259 |

### Strict Validation A/B 개선량

| Validation | ΔRMSE | ΔMAE | ΔMAPE | ΔR² |
|---|---:|---:|---:|---:|
| random_split | -196.03 | -199.81 | -1.47%p | +0.0134 |
| time_holdout_test_2026 | -868.22 | -633.29 | -2.64%p | +0.0942 |
| complex_holdout | -316.25 | -200.66 | -0.80%p | +0.0539 |
| dong_holdout | -1739.59 | -1303.32 | -6.91%p | +0.3751 |

해석:

- 네 가지 검증 모두에서 `B_spatial_extended`가 `A_baseline`보다 RMSE, MAE, MAPE가 낮고 R²가 높다.
- 특히 시간 기준 holdout과 단지 기준 holdout에서도 개선 방향이 유지되므로, random split에만 의존했다는 약점이 상당히 줄어든다.
- 동 기준 holdout은 가장 엄격한 검증인데도 공간변수 추가 효과가 크게 나타났다. 다만 test 동의 구성에 따라 변동성이 클 수 있으므로, 가능하면 여러 seed로 반복 검증하거나 교수님께 이 해석이 타당한지 확인받는 것이 좋다.

## 8. 현재 결과의 강점

1. 월세 환산 가정 없이 전세-only로 분석했다.
2. target이 `보증금(만원)`으로 명확하다.
3. MAPE까지 추가해 상대 오차를 제시할 수 있다.
4. main A/B 비교에서 모든 모델이 같은 방향으로 개선됐다.
5. VIF 기반으로 명백한 중복 변수(`동_총학교수`, `총공원면적`)를 제거했다.
6. GridSearchCV와 5-fold CV를 적용했다.
7. 시간/단지/동 holdout 검증을 통해 random split 성능의 과대평가 여부를 점검할 수 있도록 했다.

## 9. 교수님께 꼭 여쭤볼 질문

### 1. 전세-only 설계가 더 적절한지

현재는 월세 환산 가정을 제거하기 위해 전세 거래만 사용했다. 이 방식이 연구 질문에 더 적합한지 확인이 필요하다.

질문:

> 월세를 전세환산해 표본 수를 늘리는 방식보다, 전세-only로 target을 명확히 하는 방식이 본 연구 목적에 더 적절한지 궁금합니다.

### 2. Scaler 미적용이 적절한지

이번 노트북은 요청대로 scaler를 사용하지 않았다. RandomForest와 XGBoost는 큰 문제가 없지만, Ridge는 변수 scale에 민감하다.

질문:

> Tree-based model은 scaler 없이도 괜찮지만, Ridge는 scale에 민감합니다. 전세-only 실험에서도 Ridge 비교를 위해 scaler를 적용하는 버전을 별도로 보고하는 것이 좋을까요?

### 3. 공간변수 효과를 main comparison 기준으로 해석해도 되는지

main comparison에서는 Group B가 모든 모델에서 좋아졌다. 하지만 `읍면동`을 추가하면 효과가 약해진다.

질문:

> 공간변수의 추가 예측력은 `A_baseline` vs `B_spatial_extended`를 중심으로 해석하고, `읍면동` 포함 결과는 정보 중복을 보여주는 supplementary analysis로 해석해도 되는지 확인받고 싶습니다.

### 4. 다중공선성 처리 방식이 적절한지

현재는 `동_총학교수`와 `총공원면적`을 제거했다.

질문:

> VIF 결과를 바탕으로 `동_총학교수`와 `총공원면적(㎡)`을 제거하고, 초/중/고 학교수와 최대공원면적만 유지한 방식이 타당한지 궁금합니다.

### 5. Holdout 검증 결과를 어떻게 해석해야 하는지

시간 기준, 단지 기준, 동 기준 holdout에서도 `B_spatial_extended`의 개선 방향은 유지됐다. 다만 동 기준 holdout은 test 동 구성에 따라 결과가 크게 달라질 수 있다.

질문:

> 시간 기준, 단지 기준, 동 기준 holdout에서도 공간변수 추가 효과가 유지됐습니다. 이 정도면 random split 과대평가 우려를 충분히 완화했다고 볼 수 있는지, 아니면 여러 random seed로 반복한 group holdout까지 추가하는 것이 좋을지 궁금합니다.

### 6. 선행연구와의 비교 방식

선행연구와 지역, 기간, target, split 방식이 다르므로 raw metric 직접 비교는 위험하다.

질문:

> 선행연구와 RMSE/R² 수치를 직접 비교하기보다, 동일 데이터 내에서 baseline 대비 공간변수 추가 개선율을 제시하는 방식이 더 적절한지 확인받고 싶습니다.

### 7. MAPE 해석 기준

Best model의 MAPE가 약 7.31%로 나왔다.

질문:

> 전세가격 예측에서 MAPE 약 7% 수준이면 수업 프로젝트 또는 논문 초안 기준에서 충분히 양호한 성능으로 해석해도 되는지 궁금합니다.

## 10. 교수님께 보여드릴 요약 문장

> 월세 환산 가정의 영향을 제거하기 위해 전세 거래 2,486건만 사용하여 추가 실험을 진행했습니다. Target은 원래 전세 보증금인 `보증금(만원)`으로 설정했고, scaler는 적용하지 않았습니다. VIF 검토 결과 완전 중복인 `동_총학교수`와 높은 중복성을 보인 `총공원면적(㎡)`은 기본 feature set에서 제외했습니다. Main Group A/B 비교 결과, Ridge, RandomForest, XGBoost 세 모델 모두에서 공간변수를 추가한 Group B가 RMSE, MAE, MAPE를 낮추고 R²를 높였습니다. 특히 XGBoost 기준으로 Group B는 RMSE 약 1,704만원, MAE 약 1,122만원, MAPE 약 7.31%, R² 약 0.9455를 기록했습니다. 또한 시간 기준, 단지 기준, 동 기준 holdout에서도 공간변수 추가 효과가 유지되었습니다. 다만 `읍면동`을 포함하면 공간변수의 추가 효과가 약해져, 행정동 변수와 공간 파생변수 간 정보 중복 가능성이 확인됩니다. 이 해석 방향과 추가 반복 검증 필요성에 대해 피드백을 받고 싶습니다.

## 11. 현재 결론 초안

전세-only random split 기준에서는 공간 파생변수의 추가 예측력이 확인된다. 추가로 시간 기준, 단지 기준, 동 기준 holdout에서도 `B_spatial_extended`가 `A_baseline`보다 일관되게 좋은 성능을 보여, 공간 파생변수의 예측력 개선 효과가 random split에만 의존한 결과는 아닌 것으로 보인다. 다만 동 기준 holdout은 표본 구성에 따라 변동성이 클 수 있으므로, 교수님께 반복 검증 필요 여부를 확인받는 것이 좋다.
