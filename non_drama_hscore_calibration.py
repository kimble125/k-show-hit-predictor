#!/usr/bin/env python3
"""
K-Show Hit Predictor: Non-Drama H-Score Calibration
===================================================

K-예능/쇼 프로그램의 흥행 가능성을 예측하기 위한 Non-Drama H-Score
6축 가중치 캘리브레이션 스크립트입니다.

This script calibrates a Non-Drama H-Score model for Korean variety,
reality, survival, music, talk, travel/food/lifestyle, and sports shows.

Key ideas
---------
- 6 axes: cast chemistry, creator power, format power, platform/scheduling,
  pre-buzz, concept-trend fit
- Dual KPI: Landing score + Longevity score
- Optional Nielsen CSV cross-check via NIELSEN_CSV_PATH or data/ default path
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output_nondrama")
OUTPUT_DIR.mkdir(exist_ok=True)

AXIS_COLS = [
    "cast_chemistry",
    "creator_power",
    "format_power",
    "platform_scheduling",
    "pre_buzz",
    "concept_trend_fit",
]

AXIS_NAMES_KR = {
    "cast_chemistry": "출연진 케미 파워",
    "creator_power": "PD/크리에이터 파워",
    "format_power": "포맷 파워",
    "platform_scheduling": "플랫폼·편성 전략",
    "pre_buzz": "사전 화제성",
    "concept_trend_fit": "콘셉트·트렌드 적합도",
}

FORMAT_LABELS = {
    "OBS": "관찰리얼리티",
    "VAR": "버라이어티",
    "SRV": "서바이벌/경연",
    "TLK": "토크쇼",
    "MUS": "음악쇼",
    "TRL": "여행/요리/라이프",
    "SPT": "스포츠예능",
}

# Landing: 첫 3~4회 안착력 / Longevity: 시즌 지속 가능성
WEIGHTS_LANDING = {
    "cast_chemistry": 14.0,
    "creator_power": 6.0,
    "format_power": 5.0,
    "platform_scheduling": 8.0,
    "pre_buzz": 20.0,
    "concept_trend_fit": 7.0,
}

WEIGHTS_LONGEVITY = {
    "cast_chemistry": 10.0,
    "creator_power": 14.0,
    "format_power": 14.0,
    "platform_scheduling": 6.0,
    "pre_buzz": 4.0,
    "concept_trend_fit": 12.0,
}

# Calibration dataset.
# Some OTT ratings are unavailable, so brand reputation is used as the main proxy.
# [EST] means the value should be treated as an estimate for portfolio/modeling purposes.
CALIBRATION_SHOWS = [
    {
        "title": "흑백요리사: 요리 계급 전쟁",
        "platform": "Netflix",
        "year": 2024,
        "format_type": "SRV",
        "genre_desc": "요리서바이벌,경연",
        "avg_rating_pct": None,
        "brand_reputation": 7700000,
        "cast_chemistry": 8,
        "creator_power": 8,
        "format_power": 9,
        "platform_scheduling": 10,
        "pre_buzz": 7,
        "concept_trend_fit": 10,
    },
    {
        "title": "나 혼자 산다",
        "platform": "MBC",
        "year": 2024,
        "format_type": "OBS",
        "genre_desc": "관찰리얼리티,싱글라이프",
        "avg_rating_pct": 4.5,
        "brand_reputation": 6500000,
        "cast_chemistry": 8,
        "creator_power": 7,
        "format_power": 9,
        "platform_scheduling": 6,
        "pre_buzz": 6,
        "concept_trend_fit": 8,
    },
    {
        "title": "미운 우리 새끼",
        "platform": "SBS",
        "year": 2025,
        "format_type": "OBS",
        "genre_desc": "관찰리얼리티,가족",
        "avg_rating_pct": 11.3,
        "brand_reputation": 5800000,
        "cast_chemistry": 7,
        "creator_power": 6,
        "format_power": 8,
        "platform_scheduling": 7,
        "pre_buzz": 5,
        "concept_trend_fit": 7,
    },
    {
        "title": "최강야구 시즌3",
        "platform": "JTBC",
        "year": 2024,
        "format_type": "SPT",
        "genre_desc": "스포츠예능,야구",
        "avg_rating_pct": 3.02,
        "brand_reputation": 5200000,
        "cast_chemistry": 9,
        "creator_power": 7,
        "format_power": 9,
        "platform_scheduling": 6,
        "pre_buzz": 8,
        "concept_trend_fit": 9,
    },
    {
        "title": "런닝맨",
        "platform": "SBS",
        "year": 2024,
        "format_type": "VAR",
        "genre_desc": "리얼버라이어티,게임,미션",
        "avg_rating_pct": 3.63,
        "brand_reputation": 5000000,
        "cast_chemistry": 8,
        "creator_power": 6,
        "format_power": 9,
        "platform_scheduling": 7,
        "pre_buzz": 5,
        "concept_trend_fit": 5,
    },
    {
        "title": "삼시세끼",
        "platform": "tvN",
        "year": 2024,
        "format_type": "TRL",
        "genre_desc": "여행,요리,라이프,힐링",
        "avg_rating_pct": 8.86,
        "brand_reputation": 7710000,
        "cast_chemistry": 8,
        "creator_power": 9,
        "format_power": 9,
        "platform_scheduling": 7,
        "pre_buzz": 7,
        "concept_trend_fit": 8,
    },
    {
        "title": "유 퀴즈 온 더 블럭",
        "platform": "tvN",
        "year": 2024,
        "format_type": "TLK",
        "genre_desc": "토크쇼,인터뷰",
        "avg_rating_pct": 3.65,
        "brand_reputation": 4200000,
        "cast_chemistry": 7,
        "creator_power": 7,
        "format_power": 7,
        "platform_scheduling": 7,
        "pre_buzz": 5,
        "concept_trend_fit": 6,
    },
    {
        "title": "피지컬: 100 시즌2",
        "platform": "Netflix",
        "year": 2024,
        "format_type": "SRV",
        "genre_desc": "서바이벌,체력경쟁",
        "avg_rating_pct": None,
        "brand_reputation": 4500000,
        "cast_chemistry": 6,
        "creator_power": 8,
        "format_power": 8,
        "platform_scheduling": 10,
        "pre_buzz": 8,
        "concept_trend_fit": 7,
    },
    {
        "title": "SNL코리아 시즌5",
        "platform": "쿠팡플레이",
        "year": 2024,
        "format_type": "VAR",
        "genre_desc": "코미디,패러디,버라이어티",
        "avg_rating_pct": None,
        "brand_reputation": 4200000,
        "cast_chemistry": 8,
        "creator_power": 7,
        "format_power": 8,
        "platform_scheduling": 7,
        "pre_buzz": 7,
        "concept_trend_fit": 8,
    },
    {
        "title": "서진이네2",
        "platform": "tvN",
        "year": 2024,
        "format_type": "TRL",
        "genre_desc": "요리,여행,식당운영",
        "avg_rating_pct": 7.80,
        "brand_reputation": 3780395,
        "cast_chemistry": 8,
        "creator_power": 9,
        "format_power": 8,
        "platform_scheduling": 7,
        "pre_buzz": 7,
        "concept_trend_fit": 7,
    },
]


def setup_korean_font() -> None:
    """Configure a Korean-capable font when possible."""
    import platform

    system = platform.system()
    if system == "Darwin":
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Apple SD Gothic Neo"
        else:
            plt.rcParams["font.family"] = "AppleGothic"
    elif system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def build_dataframe(shows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(shows)
    df["hscore_raw"] = df[AXIS_COLS].mean(axis=1) * 10

    has_rating = df["avg_rating_pct"].notna()
    if has_rating.sum() > 0:
        rating_scaler = MinMaxScaler()
        df.loc[has_rating, "rating_norm"] = rating_scaler.fit_transform(
            df.loc[has_rating, "avg_rating_pct"].values.reshape(-1, 1)
        ).ravel()

    brand_scaler = MinMaxScaler()
    df["brand_norm"] = brand_scaler.fit_transform(
        df["brand_reputation"].values.reshape(-1, 1)
    ).ravel()

    df["combined_performance"] = df.apply(
        lambda row: row["rating_norm"] * 0.6 + row["brand_norm"] * 0.4
        if pd.notna(row.get("rating_norm"))
        else row["brand_norm"],
        axis=1,
    )
    return df


def load_nielsen_data(csv_path: str | Path) -> pd.Series | None:
    """Load optional Nielsen CSV and summarize household ratings by program."""
    csv_path = Path(csv_path).expanduser()
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ⚠️ 닐슨 CSV 파일을 찾을 수 없습니다: {csv_path}")
        return None

    required = {"metric_type", "program", "metric_value"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ⚠️ 닐슨 CSV 컬럼이 예상과 다릅니다. 누락 컬럼: {sorted(missing)}")
        return None

    household = df[df["metric_type"] == "household_rating"]
    ratings = household.groupby("program")["metric_value"].mean().sort_values(ascending=False)

    print("\n  📺 닐슨 데이터 로드 완료")
    print(f"  CSV: {csv_path}")
    print(f"  프로그램 수: {len(ratings)}")
    print("\n  시청률 상위 10:")
    for program, rating in ratings.head(10).items():
        print(f"    {program:40s} : {rating:.2f}%")
    return ratings


def correlation_analysis(df: pd.DataFrame, target: str = "combined_performance") -> dict:
    print("\n" + "=" * 60)
    print(f"📊 상관 분석: target={target}")
    print("=" * 60)

    result = {}
    for col in AXIS_COLS:
        pearson = df[col].corr(df[target])
        spearman = df[col].corr(df[target], method="spearman")
        result[col] = {"pearson": float(pearson), "spearman": float(spearman)}
        print(f"  {col:25s} | Pearson: {pearson:+.3f} | Spearman: {spearman:+.3f}")
    return result


def regression_analysis(df: pd.DataFrame, target: str = "combined_performance") -> dict:
    print("\n" + "=" * 60)
    print(f"📈 회귀 분석: target={target}")
    print("=" * 60)

    X = df[AXIS_COLS].values
    y = df[target].values
    X_scaled = StandardScaler().fit_transform(X)
    loo = LeaveOneOut()

    results = {}

    lr = LinearRegression().fit(X_scaled, y)
    results["linear"] = dict(zip(AXIS_COLS, lr.coef_))
    print(f"\n  [Linear Regression] train R²={lr.score(X_scaled, y):.3f}")

    ridge = Ridge(alpha=1.0).fit(X_scaled, y)
    results["ridge"] = dict(zip(AXIS_COLS, ridge.coef_))
    ridge_scores = cross_val_score(ridge, X_scaled, y, cv=loo, scoring="neg_mean_squared_error")
    print(f"  [Ridge] LOO neg-MSE={ridge_scores.mean():.4f}")

    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    results["random_forest"] = dict(zip(AXIS_COLS, rf.feature_importances_))

    gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42).fit(X, y)
    results["gradient_boost"] = dict(zip(AXIS_COLS, gb.feature_importances_))

    for method, vals in results.items():
        print(f"\n  [{method}]")
        for col, val in sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"    {col:25s}: {val:+.4f}")

    return results


def derive_weights(correlations: dict, regression_results: dict) -> dict:
    scores = {col: [] for col in AXIS_COLS}

    corr_vals = {col: abs(correlations[col]["spearman"]) for col in AXIS_COLS}
    corr_total = sum(corr_vals.values())
    if corr_total:
        for col in AXIS_COLS:
            scores[col].append(corr_vals[col] / corr_total)

    for method in ["linear", "ridge"]:
        vals = {col: abs(float(regression_results[method][col])) for col in AXIS_COLS}
        total = sum(vals.values())
        if total:
            for col in AXIS_COLS:
                scores[col].append(vals[col] / total)

    for method in ["random_forest", "gradient_boost"]:
        vals = regression_results[method]
        total = sum(float(vals[col]) for col in AXIS_COLS)
        if total:
            for col in AXIS_COLS:
                scores[col].append(float(vals[col]) / total)

    avg_scores = {col: float(np.mean(scores[col])) for col in AXIS_COLS}
    total = sum(avg_scores.values())
    weights = {col: round(avg_scores[col] / total * 60, 1) for col in AXIS_COLS}

    diff = round(60 - sum(weights.values()), 1)
    max_col = max(weights, key=weights.get)
    weights[max_col] = round(weights[max_col] + diff, 1)

    print("\n" + "=" * 60)
    print("⚖️  Empirical H-Score Weight Allocation")
    print("=" * 60)
    for col, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {AXIS_NAMES_KR[col]:20s}: {weight:4.1f} / 60")
    print(f"  Total: {sum(weights.values()):.1f}")
    return weights


def calculate_dual_hscore(show: dict) -> tuple[float, float, float]:
    landing = sum(show[col] * (WEIGHTS_LANDING[col] / 10) for col in AXIS_COLS)
    longevity = sum(show[col] * (WEIGHTS_LONGEVITY[col] / 10) for col in AXIS_COLS)

    landing_score = round(landing / sum(WEIGHTS_LANDING.values()) * 100, 1)
    longevity_score = round(longevity / sum(WEIGHTS_LONGEVITY.values()) * 100, 1)
    combined = round(landing_score * 0.45 + longevity_score * 0.55, 1)
    return landing_score, longevity_score, combined


def plot_weight_bar(weights: dict) -> None:
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    names = [AXIS_NAMES_KR[k] for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=values, y=names, ax=ax)
    ax.set_xlabel("Weight points out of 60")
    ax.set_title("Non-Drama H-Score Empirical Weight Distribution")
    for i, v in enumerate(values):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nd_weight_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hscore_comparison(scored: list[dict]) -> None:
    ordered = sorted(scored, key=lambda x: x["hscore"], reverse=True)
    titles = [row["title"][:18] for row in ordered]
    scores = [row["hscore"] for row in ordered]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=scores, y=titles, ax=ax)
    ax.axvline(70, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("H-Score")
    ax.set_title("K-Show Non-Drama H-Score Comparison")
    for i, v in enumerate(scores):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nd_hscore_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    weighted = []
    for _, row in df.iterrows():
        score = sum(row[col] * (weights[col] / 10) for col in AXIS_COLS)
        weighted.append(round(score / sum(weights.values()) * 100, 1))
    df = df.copy()
    df["hscore_weighted"] = weighted

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(data=df, x="hscore_weighted", y="combined_performance", hue="format_type", s=100, ax=ax)
    for _, row in df.iterrows():
        ax.annotate(row["title"][:8], (row["hscore_weighted"], row["combined_performance"]), fontsize=8)
    corr = df["hscore_weighted"].corr(df["combined_performance"], method="spearman")
    ax.set_title(f"Predicted H-Score vs Combined Performance | Spearman rho={corr:.3f}")
    ax.set_xlabel("Weighted H-Score")
    ax.set_ylabel("Combined performance, normalized")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nd_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def save_results(weights: dict, scored: list[dict], df: pd.DataFrame) -> None:
    output = {
        "weights_empirical": weights,
        "weights_landing": WEIGHTS_LANDING,
        "weights_longevity": WEIGHTS_LONGEVITY,
        "dual_kpi": "Landing 45% + Longevity 55%",
        "data_size": int(len(df)),
        "format_labels": FORMAT_LABELS,
        "calibration_shows": scored,
        "version": "1.0-nondrama",
    }
    out_path = OUTPUT_DIR / "nondrama_hscore_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  💾 결과 JSON 저장: {out_path}")


def main() -> tuple[dict, list[dict]]:
    setup_korean_font()

    print("\n📦 Step 1: 캘리브레이션 데이터 로드")
    df = build_dataframe(CALIBRATION_SHOWS)
    print(f"  데이터: {len(df)}편")
    print(f"  포맷 분포: {df['format_type'].value_counts().to_dict()}")

    print("\n📦 Step 2: 닐슨 데이터 교차 검증")
    nielsen_path = os.getenv(
        "NIELSEN_CSV_PATH",
        str(Path("data") / "nielsen_weekly_all_categories_2024_2026.csv"),
    )
    load_nielsen_data(nielsen_path)

    print("\n📦 Step 3: 상관 분석")
    correlations = correlation_analysis(df)

    df_rating = df[df["avg_rating_pct"].notna()].copy()
    if len(df_rating) >= 5:
        print("\n  [시청률 직접 상관, OTT 제외]")
        correlation_analysis(df_rating, target="avg_rating_pct")

    print("\n📦 Step 4: 회귀 분석")
    regression_results = regression_analysis(df)

    print("\n📦 Step 5: 가중치 산출")
    weights = derive_weights(correlations, regression_results)

    print("\n📦 Step 6: Dual KPI H-Score 산출")
    scored = []
    for show in CALIBRATION_SHOWS:
        landing, longevity, hscore = calculate_dual_hscore(show)
        row = dict(show)
        row.update({"hscore_landing": landing, "hscore_longevity": longevity, "hscore": hscore})
        scored.append(row)

    for i, row in enumerate(sorted(scored, key=lambda x: x["hscore"], reverse=True), start=1):
        rating = f"{row['avg_rating_pct']:.1f}%" if row.get("avg_rating_pct") else "OTT"
        print(
            f"  {i:>2}. {row['title']:<24s} | "
            f"Landing {row['hscore_landing']:>5.1f} | "
            f"Longevity {row['hscore_longevity']:>5.1f} | "
            f"H {row['hscore']:>5.1f} | {rating}"
        )

    print("\n📦 Step 7: 시각화 및 결과 저장")
    plot_weight_bar(weights)
    plot_hscore_comparison(scored)
    df_scored = plot_actual_vs_predicted(df, weights)
    save_results(weights, scored, df_scored)

    print("\n✅ 완료! output_nondrama/ 폴더에서 결과를 확인하세요.")
    return weights, scored


if __name__ == "__main__":
    main()
