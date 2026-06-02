# Final XGBoost + SHAP Experiment Design

본 문서는 회의 후 확정된 실험 방향을 정리한 것이다. 최종 Colab은 `notebooks/IML_Final_Project_Final_XGBoost_SHAP_Colab.ipynb`이다.

## 1. 최종 방향

기존처럼 Ridge, RandomForest, XGBoost를 모두 최종 결과로 보여주지 않는다. 최종 모델은 예측 성능이 가장 좋았던 XGBoost 하나로 고정한다.

연구의 초점은 다음과 같이 바꾼다.

- 모델 간 성능 비교가 아니라, XGBoost 안에서 feature set을 비교한다.
- 월세 거래는 전세환산보증금으로 변환해 표본에 포함한다.
- 공간파생변수의 의미는 SHAP으로 확인한다.
- SHAP에서 공간파생변수 중요도가 높으면 기존 공간변수 논리를 유지한다.
- SHAP에서 공간파생변수 중요도가 낮으면 공간변수 강조를 줄이고 예측 성능 중심으로 해석한다.

## 2. Target

최종 target은 `전세환산보증금(만원)`이다.

전세 거래:

```text
전세환산보증금 = 보증금(만원)
```

월세 거래:

```text
전세환산보증금 = 보증금(만원) + 월세금(만원) * 12 / 전월세전환율
```

기본 전월세전환율은 `0.065`를 사용한다.

## 3. Feature Set

| Experiment | 목적 | 포함 변수 |
|---|---|---|
| `A_baseline_no_dong` | 기본 baseline | 전용면적, 층, 건물연령, 계약연도, 계약월 |
| `B_spatial_no_dong` | 읍면동 없이 공간변수 추가 효과 확인 | A + 바다/공원/학교 변수 |
| `A_location_baseline` | 행정동 통제 baseline | A + 읍면동 |
| `B_location_spatial` | 최종 성능 모델 | A + 읍면동 + 바다/공원/학교 변수 |

최종 성능 모델은 `B_location_spatial`이다.

## 4. 제외 변수

다음 변수는 기본 feature set에서 제외한다.

| 제외 변수 | 제외 이유 |
|---|---|
| `동_총학교수` | `동_초등학교수 + 동_중학교수 + 동_고등학교수`와 완전 중복 |
| `총공원면적(㎡)` | 기존 VIF 점검에서 `최대공원면적(㎡)`, `공원수` 등과 높은 중복성 확인 |

## 5. 읍면동과 공간변수 처리

공간파생변수는 대부분 행정동 단위 집계 변수다. 따라서 `읍면동` one-hot encoding과 공간파생변수를 함께 넣으면 정보가 중복될 수 있다.

이를 해결하기 위해 두 비교를 모두 제시한다.

1. `A_baseline_no_dong` vs `B_spatial_no_dong`
   - 공간변수의 기본 추가 효과 확인

2. `A_location_baseline` vs `B_location_spatial`
   - 읍면동을 통제한 뒤에도 공간변수가 추가 가치를 갖는지 확인

XGBoost에서는 선형회귀처럼 다중공선성 때문에 모델 학습이 직접적으로 깨지지는 않는다. 다만 SHAP 중요도가 `읍면동`과 공간변수 사이에 나뉘어 해석될 수 있으므로, grouped SHAP을 함께 확인한다.

## 6. Model

최종 모델은 XGBoost Regressor 하나만 사용한다.

```python
XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    importance_type="gain",
)
```

GridSearchCV 탐색 범위:

| Parameter | Values |
|---|---|
| `n_estimators` | 300, 600 |
| `max_depth` | 4, 6 |
| `learning_rate` | 0.05, 0.1 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.9 |

Train set 내부에서 5-fold cross validation으로 최적 파라미터를 선택하고, 최종 test set에서 성능을 평가한다.

## 7. Evaluation

평가지표는 다음 네 가지다.

| Metric | 의미 |
|---|---|
| RMSE | 큰 오차에 민감한 예측 오차 |
| MAE | 평균적으로 몇 만원 정도 틀리는지 |
| MAPE | 실제 가격 대비 평균 몇 % 틀리는지 |
| R² | target 변동 설명력 |

## 8. SHAP Interpretation

SHAP은 최종 모델 `B_location_spatial` 기준으로 계산한다.

확인할 내용:

- Top feature importance
- Grouped SHAP importance
- 공간변수 그룹 비중: sea + park + school

해석 기준:

- 공간변수 SHAP 비중이 충분히 크면 공간파생변수의 중요성을 강조한다.
- 공간변수 SHAP 비중이 낮으면 예측 성능 중심으로 결론을 조정한다.
- `읍면동` 중요도가 매우 크면 행정동 위치효과가 강하다고 해석한다.

## 9. Robustness Validation

최종 Colab은 `A_location_baseline`과 `B_location_spatial`을 대상으로 다음 strict validation을 수행한다.

| Validation | 목적 |
|---|---|
| random split | 일반 80/20 split |
| time holdout | 과거 연도로 학습하고 최신 연도 테스트 |
| complex holdout | 학습에 없는 단지 테스트 |
| dong holdout | 학습에 없는 행정동 테스트 |

이 검증에서도 `B_location_spatial`이 `A_location_baseline`보다 좋은지 확인한다.

## 10. 기존 설계에서 바뀐 점

| 항목 | 기존 | 최종 |
|---|---|---|
| 모델 | Ridge, RandomForest, XGBoost 비교 | XGBoost 하나만 사용 |
| 핵심 주장 | 공간파생변수의 추가 예측력 | XGBoost 예측 성능 + SHAP 기반 공간변수 해석 |
| Target | 전세-only 또는 전세환산 실험 분리 | 전세환산보증금 main |
| 월세 행 | 전세-only에서는 제거 | 전세환산 후 포함 |
| Scaler | Ridge 때문에 논의 필요 | XGBoost만 사용하므로 scaler 미적용 |
| 읍면동 | main/supplementary로 분리 | 최종 모델에는 포함, no-dong 비교도 함께 제시 |
| 해석 | 모델별 성능 비교 | feature set 비교와 SHAP 해석 |

