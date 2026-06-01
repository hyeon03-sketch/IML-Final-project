import numpy as np
import pandas as pd

# 엑셀 데이터 파일과 분석할 시트 이름
DATA_PATH = "IML_Final_dataset.xlsx"
SHEET_NAME = "최종데이터셋_모델용"

# 데이터 불러오기
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)


def calculate_vif(data, columns):
    # 분석할 변수만 가져오고 결측치 및 무한대 제거
    X = (
        data[columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    rows = []

    # 각 변수를 나머지 변수들로 설명했을 때의 VIF 계산
    for col in columns:
        y = X[col].to_numpy()

        others = [c for c in columns if c != col]
        X_other = X[others].to_numpy()

        # 절편항 추가
        X_design = np.column_stack([np.ones(len(X_other)), X_other])

        # 선형회귀 계산
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        pred = X_design @ beta

        # R² 계산
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # VIF 계산
        vif = np.inf if r2 >= 0.999999 else 1 / (1 - r2)

        rows.append({"feature": col, "vif": vif})

    return pd.DataFrame(rows).sort_values("vif", ascending=False)


# VIF를 확인할 변수 목록
vif_cols = [
    "전용면적(㎡)",
    "층",
    "건물연령(계약기준)",
    "계약연도",
    "계약월",
    "공원수",
    "최대공원면적(㎡)",
    "총공원면적(㎡)",
    "동_초등학교수",
    "동_중학교수",
    "동_고등학교수",
]

# VIF 계산
vif_result = calculate_vif(df, vif_cols)

# 터미널 출력
print("\n=== 전체 VIF 결과 ===")
print(vif_result.to_string(index=False))

# CSV 파일 저장
vif_result.to_csv("vif_results.csv", index=False, encoding="utf-8-sig")

print("\n저장 완료: vif_results.csv")