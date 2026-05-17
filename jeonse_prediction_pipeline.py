"""
=========================================================================
포항 북구 전세 보증금 예측 — 공간변수 기여도 분석
기계학습개론 Final Project | Team 호재호
-------------------------------------------------------------------------
Pipeline:
  Dataset → Preprocessing → Feature Engineering → ML Models (3) → Eval → SHAP
Models:
  Ridge Regression / Random Forest / XGBoost
Feature Groups:
  Group A (Baseline) : 주택기본 + 시점 + 읍면동 + 바다
  Group B (Spatial)  : Group A + 공원/학교 변수
=========================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

import shap

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"  # 한글 폰트는 환경에 맞춰 교체
RND = 42
np.random.seed(RND)

# -------------------------------------------------------------------------
# 0. 경로 / 출력 폴더
# -------------------------------------------------------------------------
DATA_PATH = "IML_Final_dataset.xlsx"      # 같은 폴더에 두기
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# 1. 데이터 로드 + 월세 → 전세 환산
# -------------------------------------------------------------------------
def load_and_convert(path: str) -> pd.DataFrame:
    """
    월세 거래는 전월세 전환율을 이용해 전세보증금으로 환산한다.
        전세환산 = 월세보증금 + (월세 × 12 / 전환율)
    경북 지역의 평균 전월세 전환율(약 6.5%)을 사용.
    """
    df = pd.read_excel(path, sheet_name="최종데이터셋_모델용")

    CONV_RATE = 0.065  # 경북지역 평균 전월세 전환율
    is_wolse = df["전월세구분"] == "월세"

    df["보증금_환산(만원)"] = df["보증금(만원)"].astype(float)
    df.loc[is_wolse, "보증금_환산(만원)"] = (
        df.loc[is_wolse, "보증금(만원)"]
        + df.loc[is_wolse, "월세금(만원)"] * 12.0 / CONV_RATE
    )

    # 데이터 sanity check: 0원이거나 비정상 값 제거
    df = df[df["보증금_환산(만원)"] > 0].copy()

    print(f"[LOAD] 총 {len(df):,}건 (전세 {(df['전월세구분']=='전세').sum()}, "
          f"월세 환산 {(df['전월세구분']=='월세').sum()})")
    return df


# -------------------------------------------------------------------------
# 2. EDA — 분포 / 상관관계 / 결측·이상치 점검
# -------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    print("\n[EDA] 결측치:")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\n[EDA] 보증금_환산 기술통계:")
    print(df["보증금_환산(만원)"].describe())

    # 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["보증금_환산(만원)"], bins=60, ax=axes[0], color="#3b82f6")
    axes[0].set_title("Jeonse-equivalent deposit (raw)")
    sns.histplot(np.log1p(df["보증금_환산(만원)"]), bins=60, ax=axes[1], color="#10b981")
    axes[1].set_title("log(1 + deposit)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_target_dist.png", dpi=150)
    plt.close()

    # 상관 히트맵
    num_cols = ["보증금_환산(만원)", "전용면적(㎡)", "층", "건물연령(계약기준)",
                "공원수", "최대공원면적(㎡)", "총공원면적(㎡)", "바다여부",
                "동_초등학교수", "동_중학교수", "동_고등학교수", "동_총학교수"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_corr.png", dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# 3. Feature Group 정의
# -------------------------------------------------------------------------
GROUP_A_NUMERIC = ["전용면적(㎡)", "층", "건물연령(계약기준)", "계약연도", "계약월"]
GROUP_A_BINARY  = ["바다여부"]
GROUP_A_CATEGORICAL = ["읍면동"]

# 확장(공간) 변수 — 확장모형A 기준 (개별 학교수/공원수). 공간변수의 해석 가능성이 가장 높음
GROUP_B_EXTRA_NUMERIC = ["공원수", "동_초등학교수", "동_중학교수", "동_고등학교수"]
GROUP_B_EXTRA_BINARY  = ["공원여부"]
# (선택) 확장모형B에서 사용 — 다중공선성 점검 후 추가
GROUP_B_EXTRA_AREA    = ["최대공원면적(㎡)", "총공원면적(㎡)"]


def build_feature_frame(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "A":
        cols = GROUP_A_NUMERIC + GROUP_A_BINARY + GROUP_A_CATEGORICAL
    elif group == "B":
        cols = (GROUP_A_NUMERIC + GROUP_B_EXTRA_NUMERIC + GROUP_B_EXTRA_AREA
                + GROUP_A_BINARY + GROUP_B_EXTRA_BINARY + GROUP_A_CATEGORICAL)
    else:
        raise ValueError(group)
    X = df[cols].copy()
    return X


def make_preprocessor(numeric_cols, categorical_cols):
    """수치형 표준화 + 범주형 OneHot."""
    from sklearn.preprocessing import OneHotEncoder
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_cols),
        ],
        remainder="passthrough",   # 0/1 더미는 그대로
    )


# -------------------------------------------------------------------------
# 4. 모델 학습 + 5-Fold CV + 하이퍼파라미터 튜닝
# -------------------------------------------------------------------------
def get_models_and_grids():
    """경량 그리드 (학습 속도 우선). 최종 발표용으로는 grid를 더 넓혀도 됨."""
    return {
        "Ridge": (
            Ridge(random_state=RND),
            {"model__alpha": [1.0, 10.0, 100.0]},
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=RND, n_jobs=-1),
            {
                "model__n_estimators": [300],
                "model__max_depth":    [None, 20],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "XGBoost": (
            XGBRegressor(
                random_state=RND, n_jobs=-1, tree_method="hist",
                objective="reg:squarederror"
            ),
            {
                "model__n_estimators":  [500],
                "model__max_depth":     [4, 6],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
    }


def fit_eval(df: pd.DataFrame, group: str) -> dict:
    """주어진 feature group으로 3개 모델 모두 학습/평가."""
    print(f"\n========= [GROUP {group}] 모델 학습 시작 =========")

    X = build_feature_frame(df, group)
    y = df["보증금_환산(만원)"]

    numeric_cols = [c for c in X.columns
                    if c not in GROUP_A_CATEGORICAL + GROUP_A_BINARY + GROUP_B_EXTRA_BINARY]
    categorical_cols = GROUP_A_CATEGORICAL

    pre = make_preprocessor(numeric_cols, categorical_cols)

    # train/test split (8:2)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RND
    )

    results = {}
    for name, (estimator, grid) in get_models_and_grids().items():
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        gs = GridSearchCV(
            pipe, grid, cv=KFold(5, shuffle=True, random_state=RND),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1, verbose=0,
        )
        gs.fit(X_tr, y_tr)

        y_pred = gs.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)
        r2   = r2_score(y_te, y_pred)
        cv_rmse = -gs.best_score_

        print(f"  {name:>13s} | CV-RMSE {cv_rmse:8.1f}  "
              f"Test RMSE {rmse:8.1f}  MAE {mae:8.1f}  R² {r2:.4f}")

        results[name] = {
            "best_params": gs.best_params_,
            "cv_rmse": cv_rmse,
            "test_rmse": rmse, "test_mae": mae, "test_r2": r2,
            "model": gs.best_estimator_,
            "X_test": X_te, "y_test": y_te, "y_pred": y_pred,
        }
    return results


# -------------------------------------------------------------------------
# 5. 결과 비교 표 / 시각화 / SHAP
# -------------------------------------------------------------------------
def make_summary_table(res_a: dict, res_b: dict) -> pd.DataFrame:
    rows = []
    for name in res_a:
        a, b = res_a[name], res_b[name]
        rows.append({
            "Model": name,
            "RMSE_A": a["test_rmse"], "RMSE_B": b["test_rmse"],
            "ΔRMSE":  b["test_rmse"] - a["test_rmse"],
            "MAE_A":  a["test_mae"],  "MAE_B": b["test_mae"],
            "ΔMAE":   b["test_mae"]  - a["test_mae"],
            "R2_A":   a["test_r2"],   "R2_B": b["test_r2"],
            "ΔR2":    b["test_r2"]   - a["test_r2"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/summary_metrics.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_pred_vs_true(res_b: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, r) in zip(axes, res_b.items()):
        ax.scatter(r["y_test"], r["y_pred"], s=8, alpha=0.4, color="#3b82f6")
        lim = max(r["y_test"].max(), r["y_pred"].max())
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_title(f"{name}  (R²={r['test_r2']:.3f})")
        ax.set_xlabel("Actual (만원)"); ax.set_ylabel("Predicted (만원)")
    plt.suptitle("Prediction vs Actual — Group B (Spatial)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pred_vs_actual.png", dpi=150)
    plt.close()


def shap_for_xgb(res_b: dict, df: pd.DataFrame) -> None:
    """XGBoost SHAP — Group B에서 공간변수 기여도 가시화."""
    r = res_b["XGBoost"]
    pipe = r["model"]
    pre  = pipe.named_steps["pre"]
    xgb  = pipe.named_steps["model"]

    X_te_trans = pre.transform(r["X_test"])
    feat_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(X_te_trans)

    # SHAP summary plot
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X_te_trans, feature_names=feat_names,
                      max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_summary_xgb_groupB.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # 평균 |SHAP| 기준 top 변수
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).head(20)
    imp.to_csv(f"{OUT_DIR}/shap_top20_features.csv",
               index=False, encoding="utf-8-sig")


# -------------------------------------------------------------------------
# 6. main
# -------------------------------------------------------------------------
def main():
    df = load_and_convert(DATA_PATH)
    run_eda(df)

    res_a = fit_eval(df, "A")
    res_b = fit_eval(df, "B")

    summary = make_summary_table(res_a, res_b)
    print("\n========= 모델 성능 비교 (Group A vs Group B) =========")
    print(summary.to_string(index=False, float_format="%.2f"))

    plot_pred_vs_true(res_b)
    shap_for_xgb(res_b, df)

    # 모델별 best params 저장
    bp_rows = []
    for g, res in [("A", res_a), ("B", res_b)]:
        for name, r in res.items():
            bp_rows.append({"group": g, "model": name,
                            "best_params": r["best_params"],
                            "cv_rmse": r["cv_rmse"]})
    pd.DataFrame(bp_rows).to_csv(
        f"{OUT_DIR}/best_params.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n[DONE] 결과 파일이 ./{OUT_DIR}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
