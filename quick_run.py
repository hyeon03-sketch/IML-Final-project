"""
Quick demo run — 고정 하이퍼파라미터로 빠르게 결과 산출.
실제 발표용 코드는 jeonse_prediction_pipeline.py 사용.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RND = 42

DATA = "/sessions/practical-pensive-knuth/mnt/기계학습개론 final project/IML_Final_dataset.xlsx"
OUT  = "/sessions/practical-pensive-knuth/mnt/기계학습개론 final project/outputs"
os.makedirs(OUT, exist_ok=True)

# 1. 로드 + 월세→전세 환산
df = pd.read_excel(DATA, sheet_name="최종데이터셋_모델용")
CONV = 0.065
mask = df["전월세구분"] == "월세"
df["보증금_환산"] = df["보증금(만원)"].astype(float)
df.loc[mask, "보증금_환산"] = df.loc[mask, "보증금(만원)"] + df.loc[mask, "월세금(만원)"]*12.0/CONV
df = df[df["보증금_환산"] > 0].copy()
print(f"[LOAD] N={len(df):,}")

# 2. Feature 정의
A_NUM = ["전용면적(㎡)","층","건물연령(계약기준)","계약연도","계약월"]
A_BIN = ["바다여부"]
A_CAT = ["읍면동"]
B_NUM_EXTRA = ["공원수","최대공원면적(㎡)","총공원면적(㎡)","동_초등학교수","동_중학교수","동_고등학교수"]
B_BIN_EXTRA = ["공원여부","동_초중고모두있음여부"]

def cols_for(group):
    if group == "A": return A_NUM + A_BIN + A_CAT
    return A_NUM + B_NUM_EXTRA + A_BIN + B_BIN_EXTRA + A_CAT

def build_pre(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], remainder="passthrough")

# 3. 학습/평가
def run(group):
    cols = cols_for(group)
    X = df[cols]; y = df["보증금_환산"]
    num_cols = [c for c in cols if c not in A_CAT + A_BIN + B_BIN_EXTRA]
    pre = build_pre(num_cols, A_CAT)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RND)

    models = {
        "Ridge":         Ridge(alpha=10.0, random_state=RND),
        "RandomForest":  RandomForestRegressor(n_estimators=300, max_depth=20,
                                               min_samples_leaf=2, n_jobs=-1, random_state=RND),
        "XGBoost":       XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                                      subsample=0.9, colsample_bytree=0.9,
                                      tree_method="hist", random_state=RND, n_jobs=-1),
    }
    out = {}
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("model", m)])
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        out[name] = {
            "rmse": float(np.sqrt(mean_squared_error(yte, p))),
            "mae":  float(mean_absolute_error(yte, p)),
            "r2":   float(r2_score(yte, p)),
            "pipe": pipe, "Xte": Xte, "yte": yte, "pred": p,
        }
        print(f"[{group}] {name:>13s}  RMSE={out[name]['rmse']:8.1f}  "
              f"MAE={out[name]['mae']:7.1f}  R²={out[name]['r2']:.4f}")
    return out

print("\n=== Group A (Baseline) ===")
res_a = run("A")
print("\n=== Group B (Spatial) ===")
res_b = run("B")

# 4. summary 테이블
rows = []
for name in res_a:
    a, b = res_a[name], res_b[name]
    rows.append({
        "Model": name,
        "RMSE_A": a["rmse"], "RMSE_B": b["rmse"], "ΔRMSE": b["rmse"]-a["rmse"],
        "MAE_A": a["mae"],   "MAE_B": b["mae"],   "ΔMAE":  b["mae"]-a["mae"],
        "R2_A": a["r2"],     "R2_B": b["r2"],     "ΔR2":   b["r2"]-a["r2"],
    })
summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT}/summary_metrics.csv", index=False, encoding="utf-8-sig")
print("\n=== Summary ===")
print(summary.to_string(index=False, float_format="%.2f"))

# 5. Pred vs Actual plot (Group B)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, r) in zip(axes, res_b.items()):
    ax.scatter(r["yte"], r["pred"], s=8, alpha=0.4, color="#3b82f6")
    lim = max(r["yte"].max(), r["pred"].max())
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_title(f"{name}  R²={r['r2']:.3f}")
    ax.set_xlabel("Actual (만원)"); ax.set_ylabel("Predicted (만원)")
plt.suptitle("Prediction vs Actual — Group B")
plt.tight_layout()
plt.savefig(f"{OUT}/pred_vs_actual.png", dpi=140)
plt.close()

# 6. XGBoost feature importance (built-in gain)
xgb_pipe = res_b["XGBoost"]["pipe"]
pre_b = xgb_pipe.named_steps["pre"]
xgb   = xgb_pipe.named_steps["model"]
fnames = pre_b.get_feature_names_out()
imp = pd.DataFrame({"feature": fnames, "gain": xgb.feature_importances_}) \
        .sort_values("gain", ascending=False)
imp.to_csv(f"{OUT}/xgb_feature_importance.csv", index=False, encoding="utf-8-sig")

top = imp.head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top["feature"], top["gain"], color="#10b981")
plt.title("XGBoost Feature Importance — Group B (top 15)")
plt.tight_layout()
plt.savefig(f"{OUT}/xgb_top_features.png", dpi=140)
plt.close()

print("\n=== XGBoost Top 15 (gain) ===")
print(imp.head(15).to_string(index=False))

# JSON dump
with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
    json.dump({
        "group_A": {k: {kk: vv for kk, vv in v.items() if kk in ["rmse","mae","r2"]}
                    for k, v in res_a.items()},
        "group_B": {k: {kk: vv for kk, vv in v.items() if kk in ["rmse","mae","r2"]}
                    for k, v in res_b.items()},
    }, f, ensure_ascii=False, indent=2)

print(f"\n[DONE] outputs at {OUT}")
