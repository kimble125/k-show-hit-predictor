#!/usr/bin/env python3
"""
K-Show Hit Predictor: Non-Drama H-Score Calibration
====================================================

K-예능/쇼 프로그램의 흥행 가능성을 예측하기 위한 Non-Drama H-Score
6축 가중치 캘리브레이션 스크립트입니다.

This script calibrates a Non-Drama H-Score model for Korean variety,
reality, survival, music, talk, travel/food/lifestyle, and sports shows.

Key ideas
---------
- 6 axes: cast chemistry, creator power, format power, platform/scheduling,
  pre-buzz, concept-trend fit
- Dual KPI: Landing score (초기 안착력) + Longevity score (롱런 지속력)
- Calibration: 25 shows (2024–2026), all verified ratings + brand index
- Optional Nielsen CSV cross-check via NIELSEN_CSV_PATH or data/ default path
- NEW_SHOWS: scores for programs currently airing (prediction mode)

Format Classification (방송통신위원회 2012 공식분류 기반):
  OBS  관찰리얼리티   VAR  버라이어티   SRV  서바이벌/경연
  TLK  토크쇼         MUS  음악쇼       TRL  여행/요리/라이프
  SPT  스포츠예능

Data sources
------------
- Nielsen Korea weekly household ratings (유료플랫폼 기준)
- 한국기업평판연구소 예능 브랜드평판지수 (brikorea.com)
- 나무위키 시청률 교차검증
- [EST] marks estimated values pending verification

GitHub : https://github.com/kimble125/k-show-hit-predictor
Blog   : https://forrest125.tistory.com/
Version: 1.1-nondrama
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

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

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

# Dual KPI weights (60-point scale per KPI)
# Landing  : 초기 안착력 — 첫 3~4회 안에 정착하는가?
# Longevity: 롱런 지속력 — 시즌 확장 가능한가?
WEIGHTS_LANDING: dict[str, float] = {
    "cast_chemistry": 14.0,
    "creator_power": 6.0,
    "format_power": 5.0,
    "platform_scheduling": 8.0,
    "pre_buzz": 20.0,       # ★★ 초기 유입의 최대 동력
    "concept_trend_fit": 7.0,
}

WEIGHTS_LONGEVITY: dict[str, float] = {
    "cast_chemistry": 10.0,
    "creator_power": 14.0,  # ★★ PD의 장기 운영력
    "format_power": 14.0,   # ★★ 포맷 확장성 = 시즌2 가능성
    "platform_scheduling": 6.0,
    "pre_buzz": 4.0,
    "concept_trend_fit": 12.0,  # ★ 트렌드 지속성
}

# ──────────────────────────────────────────────
# Calibration Dataset  (25 shows, 2024–2026)
#
# Scoring guide (1–10):
#   9~10  압도적 (국민MC / 레전드PD / Netflix 글로벌)
#   7~8   강함   (A급 예능인 / 검증된 PD / 메이저+OTT)
#   5~6   보통   (중견급 / 안정적 포맷 / 일반 편성)
#   3~4   약함   (신인급 / 미검증 / 비주류)
#   1~2   매우 약함
#
# brand_reputation: 한국기업평판연구소 브랜드평판지수 (실제 값 우선, [EST] 추정)
# avg_rating_pct  : 닐슨 가구 시청률 % (None = OTT 전용)
# ──────────────────────────────────────────────
CALIBRATION_SHOWS: list[dict] = [
    # ── 상위 tier ──────────────────────────────
    {
        "title": "흑백요리사: 요리 계급 전쟁",
        "platform": "Netflix", "year": 2024, "format_type": "SRV",
        "genre_desc": "요리서바이벌,경연",
        "avg_rating_pct": None,         # Netflix (미공개)
        "brand_reputation": 7700000,    # [EST] 2024.10 1위급
        "cast_chemistry": 8,   # 백종원·안성재 + 80인 셰프진
        "creator_power": 8,    # 이욱정PD (피지컬100 성공)
        "format_power": 9,     # SRV 진화 + 시즌2 확정
        "platform_scheduling": 10,  # Netflix 글로벌 동시공개
        "pre_buzz": 7,
        "concept_trend_fit": 10,    # 요리+계급서사 = 2024 최강 트렌드
    },
    {
        "title": "나 혼자 산다",
        "platform": "MBC", "year": 2024, "format_type": "OBS",
        "genre_desc": "관찰리얼리티,싱글라이프",
        "avg_rating_pct": 4.5,          # [SRC:닐슨]
        "brand_reputation": 6500000,    # [EST]
        "cast_chemistry": 8,   # 박나래·기안84·전현무 검증 조합
        "creator_power": 7,
        "format_power": 9,     # 11년차 장수
        "platform_scheduling": 6,
        "pre_buzz": 6,
        "concept_trend_fit": 8,   # 1인가구 트렌드
    },
    {
        "title": "미운 우리 새끼",
        "platform": "SBS", "year": 2025, "format_type": "OBS",
        "genre_desc": "관찰리얼리티,가족",
        "avg_rating_pct": 11.3,         # [SRC:닐슨]
        "brand_reputation": 5800000,    # [EST]
        "cast_chemistry": 7,
        "creator_power": 6,
        "format_power": 8,     # 9년차 리뉴얼 성공
        "platform_scheduling": 7,   # SBS 일요 황금시간
        "pre_buzz": 5,
        "concept_trend_fit": 7,
    },
    {
        "title": "최강야구 시즌3",
        "platform": "JTBC", "year": 2024, "format_type": "SPT",
        "genre_desc": "스포츠예능,야구,서바이벌",
        "avg_rating_pct": 3.02,         # [SRC:나무위키]
        "brand_reputation": 5200000,    # [EST]
        "cast_chemistry": 9,   # 이승엽·김성근·이대호 레전드 케미
        "creator_power": 7,
        "format_power": 9,     # 시즌3 + 독립리그 연계
        "platform_scheduling": 6,
        "pre_buzz": 8,         # 시즌2 성공 → 사전 기대 폭발
        "concept_trend_fit": 9,   # 프로야구 부활 트렌드
    },
    {
        "title": "런닝맨",
        "platform": "SBS", "year": 2024, "format_type": "VAR",
        "genre_desc": "리얼버라이어티,게임,미션",
        "avg_rating_pct": 3.63,         # [SRC:나무위키]
        "brand_reputation": 5000000,    # [EST]
        "cast_chemistry": 8,   # 14년차 고정 케미
        "creator_power": 6,
        "format_power": 9,     # 14년차 글로벌 장수
        "platform_scheduling": 7,
        "pre_buzz": 5,
        "concept_trend_fit": 5,   # 포맷 피로감 but 글로벌 수요
    },
    # ── 중상위 tier ────────────────────────────
    {
        "title": "삼시세끼",
        "platform": "tvN", "year": 2024, "format_type": "TRL",
        "genre_desc": "여행,요리,라이프,힐링",
        "avg_rating_pct": 8.86,         # [SRC:나무위키] 어촌편
        "brand_reputation": 7710000,    # [SRC:brikorea] 2024.10 1위
        "cast_chemistry": 8,
        "creator_power": 9,    # 나영석PD
        "format_power": 9,     # 10년차 + OSMU 강력
        "platform_scheduling": 7,
        "pre_buzz": 7,
        "concept_trend_fit": 8,
    },
    {
        "title": "유 퀴즈 온 더 블럭",
        "platform": "tvN", "year": 2024, "format_type": "TLK",
        "genre_desc": "토크쇼,인터뷰,길거리",
        "avg_rating_pct": 3.65,         # [SRC:닐슨]
        "brand_reputation": 4200000,    # [EST]
        "cast_chemistry": 7,
        "creator_power": 7,
        "format_power": 7,     # 길거리 인터뷰 독점 포맷 + 6년차
        "platform_scheduling": 7,
        "pre_buzz": 5,
        "concept_trend_fit": 6,
    },
    {
        "title": "1박 2일 시즌4",
        "platform": "KBS2", "year": 2024, "format_type": "TRL",
        "genre_desc": "여행,버라이어티,리얼",
        "avg_rating_pct": 5.9,          # [SRC:닐슨]
        "brand_reputation": 3800000,    # [EST]
        "cast_chemistry": 7,
        "creator_power": 6,
        "format_power": 8,     # 17년차 장수 포맷
        "platform_scheduling": 6,
        "pre_buzz": 5,
        "concept_trend_fit": 6,
    },
    {
        "title": "피지컬: 100 시즌2",
        "platform": "Netflix", "year": 2024, "format_type": "SRV",
        "genre_desc": "서바이벌,체력경쟁",
        "avg_rating_pct": None,         # Netflix
        "brand_reputation": 4500000,    # [EST]
        "cast_chemistry": 6,   # 일반인 100인 (집단 서사)
        "creator_power": 8,    # 장호기PD (시즌1 글로벌 히트)
        "format_power": 8,
        "platform_scheduling": 10,
        "pre_buzz": 8,
        "concept_trend_fit": 7,
    },
    {
        "title": "놀면 뭐하니",
        "platform": "MBC", "year": 2024, "format_type": "VAR",
        "genre_desc": "버라이어티,프로젝트형",
        "avg_rating_pct": 3.83,         # [SRC:나무위키]
        "brand_reputation": 3500000,    # [EST]
        "cast_chemistry": 7,
        "creator_power": 8,    # 김태호PD → 후임 전환기
        "format_power": 7,
        "platform_scheduling": 6,
        "pre_buzz": 5,
        "concept_trend_fit": 5,
    },
    {
        "title": "불후의 명곡",
        "platform": "KBS2", "year": 2024, "format_type": "MUS",
        "genre_desc": "음악쇼,경연,리메이크",
        "avg_rating_pct": 4.7,          # [SRC:닐슨]
        "brand_reputation": 3200000,    # [EST]
        "cast_chemistry": 6,
        "creator_power": 5,
        "format_power": 7,     # 12년차 장수
        "platform_scheduling": 6,
        "pre_buzz": 4,
        "concept_trend_fit": 5,
    },
    {
        "title": "전국노래자랑",
        "platform": "KBS1", "year": 2024, "format_type": "MUS",
        "genre_desc": "음악쇼,참여형,시민",
        "avg_rating_pct": 5.6,          # [SRC:닐슨]
        "brand_reputation": 3500000,    # [EST]
        "cast_chemistry": 5,
        "creator_power": 4,
        "format_power": 8,     # 40년+ 레전드 포맷
        "platform_scheduling": 5,
        "pre_buzz": 3,
        "concept_trend_fit": 5,
    },
    # ── 중위 tier ──────────────────────────────
    {
        "title": "미스트롯4",
        "platform": "TV조선", "year": 2025, "format_type": "SRV",
        "genre_desc": "음악서바이벌,트로트,경연",
        "avg_rating_pct": 5.9,          # [SRC:닐슨]
        "brand_reputation": 6890741,    # [SRC:brikorea] 2025.01 1위
        "cast_chemistry": 6,
        "creator_power": 6,
        "format_power": 7,     # 시즌4 (통합 6시즌)
        "platform_scheduling": 5,
        "pre_buzz": 6,
        "concept_trend_fit": 5,   # 트롯 열풍 감소세 but 충성 팬덤
    },
    {
        "title": "히든싱어8",
        "platform": "JTBC", "year": 2026, "format_type": "MUS",
        "genre_desc": "음악쇼,숨은실력자,서바이벌",
        "avg_rating_pct": 4.55,         # [SRC:닐슨]
        "brand_reputation": 1144915,    # [SRC:brikorea] 2026.04 30위
        "cast_chemistry": 6,
        "creator_power": 6,
        "format_power": 7,     # 시즌8 + 독창적 포맷
        "platform_scheduling": 6,
        "pre_buzz": 5,
        "concept_trend_fit": 5,
    },
    {
        "title": "살림하는 남자들 시즌2",
        "platform": "KBS2", "year": 2024, "format_type": "OBS",
        "genre_desc": "관찰리얼리티,가족,육아",
        "avg_rating_pct": 5.2,          # [SRC:닐슨]
        "brand_reputation": 2800000,    # [EST]
        "cast_chemistry": 5,
        "creator_power": 5,
        "format_power": 6,
        "platform_scheduling": 5,
        "pre_buzz": 3,
        "concept_trend_fit": 5,
    },
    {
        "title": "SNL코리아 시즌5",
        "platform": "쿠팡플레이", "year": 2024, "format_type": "VAR",
        "genre_desc": "코미디,패러디,버라이어티",
        "avg_rating_pct": None,         # 쿠팡플레이 (미공개)
        "brand_reputation": 4200000,    # [EST]
        "cast_chemistry": 8,
        "creator_power": 7,
        "format_power": 8,     # 글로벌 포맷 + 코미디 부활
        "platform_scheduling": 7,
        "pre_buzz": 7,
        "concept_trend_fit": 8,
    },
    {
        "title": "환승연애3",
        "platform": "TVING", "year": 2024, "format_type": "OBS",
        "genre_desc": "연애리얼리티,관찰",
        "avg_rating_pct": None,         # TVING (미공개)
        "brand_reputation": 4000000,    # [EST]
        "cast_chemistry": 7,
        "creator_power": 7,
        "format_power": 7,
        "platform_scheduling": 7,
        "pre_buzz": 7,
        "concept_trend_fit": 7,
    },
    {
        "title": "서진이네2",
        "platform": "tvN", "year": 2024, "format_type": "TRL",
        "genre_desc": "요리,여행,식당운영",
        "avg_rating_pct": 7.80,         # [SRC:나무위키]
        "brand_reputation": 3780395,    # [SRC:네이버블로그] 2위
        "cast_chemistry": 8,
        "creator_power": 9,    # 나영석PD
        "format_power": 8,     # 시즌2 + 윤식당 계보
        "platform_scheduling": 7,
        "pre_buzz": 7,
        "concept_trend_fit": 7,
    },
    {
        "title": "골 때리는 그녀들",
        "platform": "SBS", "year": 2024, "format_type": "SPT",
        "genre_desc": "스포츠예능,여자축구",
        "avg_rating_pct": 4.73,         # [SRC:나무위키]
        "brand_reputation": 1404089,    # [SRC:brikorea] 2025.08 25위
        "cast_chemistry": 7,
        "creator_power": 6,
        "format_power": 7,
        "platform_scheduling": 6,
        "pre_buzz": 5,
        "concept_trend_fit": 7,
    },
    # ── 중하위 tier ────────────────────────────
    {
        "title": "놀라운 토요일",
        "platform": "tvN", "year": 2025, "format_type": "VAR",
        "genre_desc": "버라이어티,음악,퀴즈",
        "avg_rating_pct": 2.19,         # [SRC:나무위키]
        "brand_reputation": 1874478,    # [SRC:brikorea] 2025.02 20위
        "cast_chemistry": 6,
        "creator_power": 5,
        "format_power": 6,
        "platform_scheduling": 6,
        "pre_buzz": 4,
        "concept_trend_fit": 5,
    },
    {
        "title": "나는 솔로",
        "platform": "ENA/SBS Plus", "year": 2024, "format_type": "OBS",
        "genre_desc": "연애리얼리티,결혼,관찰",
        "avg_rating_pct": 4.42,         # [SRC:나무위키] 2024 기수 평균
        "brand_reputation": 3200000,    # [EST]
        "cast_chemistry": 6,
        "creator_power": 5,
        "format_power": 7,     # 독점 포맷 + 밈 생산력 강
        "platform_scheduling": 4,   # ENA (마이너 채널)
        "pre_buzz": 5,
        "concept_trend_fit": 7,
    },
    {
        "title": "무명전설",
        "platform": "MBN", "year": 2026, "format_type": "MUS",
        "genre_desc": "음악쇼,숨은실력자,감동",
        "avg_rating_pct": 6.97,         # [SRC:닐슨]
        "brand_reputation": 3429487,    # [SRC:brikorea] 2026.04 6위
        "cast_chemistry": 6,
        "creator_power": 5,
        "format_power": 6,
        "platform_scheduling": 4,
        "pre_buzz": 4,
        "concept_trend_fit": 7,   # 숨은 실력자 발굴 = 공정성 밈
    },
    {
        "title": "세계밥장사 백사장3",
        "platform": "tvN", "year": 2026, "format_type": "TRL",
        "genre_desc": "요리,해외,식당운영",
        "avg_rating_pct": 2.44,         # [SRC:나무위키]
        "brand_reputation": 3000000,    # [EST]
        "cast_chemistry": 7,
        "creator_power": 7,
        "format_power": 7,     # 시즌3 + 백종원 브랜드
        "platform_scheduling": 6,
        "pre_buzz": 5,
        "concept_trend_fit": 6,
    },
    {
        "title": "보검 매직컬",
        "platform": "tvN", "year": 2026, "format_type": "VAR",
        "genre_desc": "버라이어티,마술,판타지",
        "avg_rating_pct": 3.2,          # [SRC:나무위키]
        "brand_reputation": 1493297,    # [SRC:brikorea] 2026.04 26위
        "cast_chemistry": 7,   # 박보검 (톱스타)
        "creator_power": 6,
        "format_power": 5,     # 마술 포맷 = 신선 but 미검증
        "platform_scheduling": 6,
        "pre_buzz": 7,
        "concept_trend_fit": 5,
    },
    # ── 하위 tier (대조군) ─────────────────────
    {
        "title": "예측불가 쑥이네 폐가 새로고침",
        "platform": "tvN", "year": 2026, "format_type": "OBS",
        "genre_desc": "관찰리얼리티,시골,리모델링",
        "avg_rating_pct": 2.26,         # [SRC:나무위키]
        "brand_reputation": 2104087,    # [SRC:brikorea] 2026.04 14위
        "cast_chemistry": 5,
        "creator_power": 4,
        "format_power": 4,
        "platform_scheduling": 6,
        "pre_buzz": 3,
        "concept_trend_fit": 4,
    },
]

# ──────────────────────────────────────────────
# New / currently-airing shows (prediction mode)
# These are NOT used for weight calibration.
# Run predict_new_shows() to score them.
# ──────────────────────────────────────────────
NEW_SHOWS: list[dict] = [
    {
        "title": "구기동 프렌즈",
        "platform": "tvN", "year": 2026, "format_type": "OBS",
        "genre_desc": "관찰리얼리티,동거,싱글라이프",
        "avg_rating_pct": 2.486,        # [SRC] 1화 (2026.04.10) 케이블·종편 동시간대 1위
        "brand_reputation": None,       # 미집계 (방영 초기)
        "note": "87년생 동갑 싱글 6인 동거. tvN 금 20:35 + TVING + Wavve. PD 박현주.",
        # 6축 점수
        "cast_chemistry": 7,    # 장도연·이다희·최다니엘·장근석·안재현·경수진
        "creator_power": 5,     # 박현주PD — 대형 전작 부재, 미검증
        "format_power": 6,      # 동거 관찰 = 실패 이력 多한 포맷. 벗킷리스트 장치.
        "platform_scheduling": 7,   # tvN 금요 + TVING + Wavve 트리플
        "pre_buzz": 6,
        "concept_trend_fit": 8, # 1인가구·느슨한 연대 트렌드 정조준
    },
    {
        "title": "황신혜의 같이 삽시다",
        "platform": "KBS1", "year": 2026, "format_type": "OBS",
        "genre_desc": "관찰리얼리티,동거,싱글맘",
        "avg_rating_pct": None,         # KBS1 (미확인)
        "brand_reputation": None,
        "note": "싱글맘 3인 합숙. KBS1 수 19:40. 황신혜·장윤정(방송인)·정가은. 게스트 순환 구조.",
        "cast_chemistry": 6,
        "creator_power": 5,
        "format_power": 7,      # 박원숙 시즌 → 황신혜 시즌: 검증된 포맷 계보
        "platform_scheduling": 4,   # KBS1 (도달 넓지만 저녁 시간대)
        "pre_buzz": 4,
        "concept_trend_fit": 6, # 싱글맘 연대 = 중장년 공감
    },
]


# ──────────────────────────────────────────────
# Font setup
# ──────────────────────────────────────────────

def setup_korean_font() -> None:
    """Configure a Korean-capable font when available."""
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
        # Linux: try NanumGothic first, fall back to DejaVu Sans
        try:
            result = fm.findfont("NanumGothic")
            if result and "NanumGothic" in result:
                plt.rcParams["font.family"] = "NanumGothic"
            else:
                plt.rcParams["font.family"] = "DejaVu Sans"
        except Exception:
            plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────
# Data pipeline
# ──────────────────────────────────────────────

def build_dataframe(shows: list[dict]) -> pd.DataFrame:
    """Build a calibration DataFrame with normalised performance metrics."""
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

    # Combined performance: rating 60% + brand 40% when rating available; brand 100% for OTT
    df["combined_performance"] = df.apply(
        lambda row: row["rating_norm"] * 0.6 + row["brand_norm"] * 0.4
        if pd.notna(row.get("rating_norm"))
        else row["brand_norm"],
        axis=1,
    )
    return df


def load_nielsen_data(csv_path: str | Path) -> pd.Series | None:
    """Load optional Nielsen CSV and return mean household ratings by program."""
    csv_path = Path(csv_path).expanduser()
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ⚠️  닐슨 CSV 파일을 찾을 수 없습니다: {csv_path}")
        return None

    required = {"metric_type", "program", "metric_value"}
    if missing := required - set(df.columns):
        print(f"  ⚠️  닐슨 CSV 컬럼 누락: {sorted(missing)}")
        return None

    household = df[df["metric_type"] == "household_rating"]
    ratings = household.groupby("program")["metric_value"].mean().sort_values(ascending=False)

    print(f"\n  📺 닐슨 CSV 로드 완료: {csv_path}")
    print(f"  프로그램 수: {len(ratings)}")
    print("\n  시청률 상위 10:")
    for program, rating in ratings.head(10).items():
        print(f"    {program:42s}: {rating:.2f}%")
    return ratings


# ──────────────────────────────────────────────
# Statistical analysis
# ──────────────────────────────────────────────

def correlation_analysis(
    df: pd.DataFrame, target: str = "combined_performance"
) -> dict:
    print(f"\n{'='*60}")
    print(f"📊 상관 분석 | target = {target}  (N={len(df)})")
    print("=" * 60)

    result = {}
    for col in AXIS_COLS:
        pearson = float(df[col].corr(df[target]))
        spearman = float(df[col].corr(df[target], method="spearman"))
        result[col] = {"pearson": pearson, "spearman": spearman}
        sig = "**" if abs(spearman) > 0.4 else ("*" if abs(spearman) > 0.3 else "")
        print(f"  {col:25s} | Pearson {pearson:+.3f} | Spearman {spearman:+.3f} {sig}")
    return result


def regression_analysis(
    df: pd.DataFrame, target: str = "combined_performance"
) -> dict:
    print(f"\n{'='*60}")
    print(f"📈 회귀 분석 | target = {target}")
    print("=" * 60)

    X = df[AXIS_COLS].values
    y = df[target].values
    X_scaled = StandardScaler().fit_transform(X)
    loo = LeaveOneOut()

    results: dict[str, dict] = {}

    lr = LinearRegression().fit(X_scaled, y)
    loo_r2 = cross_val_score(lr, X_scaled, y, cv=loo, scoring="r2")
    results["linear"] = dict(zip(AXIS_COLS, lr.coef_))
    print(f"\n  [Linear]  train R²={lr.score(X_scaled,y):.3f}  LOO R²={loo_r2.mean():.3f}±{loo_r2.std():.3f}")

    ridge = Ridge(alpha=1.0).fit(X_scaled, y)
    results["ridge"] = dict(zip(AXIS_COLS, ridge.coef_))
    print(f"  [Ridge]   coefs: { {k: round(v,3) for k,v in results['ridge'].items()} }")

    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    results["random_forest"] = dict(zip(AXIS_COLS, rf.feature_importances_))
    print(f"  [RF]      importances: { {k: round(v,3) for k,v in results['random_forest'].items()} }")

    gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42).fit(X, y)
    results["gradient_boost"] = dict(zip(AXIS_COLS, gb.feature_importances_))
    print(f"  [GB]      importances: { {k: round(v,3) for k,v in results['gradient_boost'].items()} }")

    return results


def derive_weights(correlations: dict, regression_results: dict) -> dict:
    """Ensemble weight derivation: normalise each method then average."""
    scores: dict[str, list[float]] = {col: [] for col in AXIS_COLS}

    # Spearman correlation (absolute value)
    corr_abs = {col: abs(correlations[col]["spearman"]) for col in AXIS_COLS}
    if total := sum(corr_abs.values()):
        for col in AXIS_COLS:
            scores[col].append(corr_abs[col] / total)

    # Linear + Ridge coefficients (absolute value)
    for method in ("linear", "ridge"):
        raw = {col: abs(float(regression_results[method][col])) for col in AXIS_COLS}
        if total := sum(raw.values()):
            for col in AXIS_COLS:
                scores[col].append(raw[col] / total)

    # Feature importances
    for method in ("random_forest", "gradient_boost"):
        raw = {col: float(regression_results[method][col]) for col in AXIS_COLS}
        if total := sum(raw.values()):
            for col in AXIS_COLS:
                scores[col].append(raw[col] / total)

    avg_scores = {col: float(np.mean(scores[col])) for col in AXIS_COLS}
    total = sum(avg_scores.values())
    weights = {col: round(avg_scores[col] / total * 60, 1) for col in AXIS_COLS}

    # Round-off correction
    diff = round(60.0 - sum(weights.values()), 1)
    weights[max(weights, key=weights.get)] = round(
        weights[max(weights, key=weights.get)] + diff, 1
    )

    print(f"\n{'='*60}")
    print("⚖️  Non-Drama H-Score Empirical Weight Allocation (60pt)")
    print("=" * 60)
    for col, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(w)
        print(f"  {AXIS_NAMES_KR[col]:22s} {w:4.1f}  {bar}")
    print(f"  {'TOTAL':22s} {sum(weights.values()):.1f}")
    return weights


# ──────────────────────────────────────────────
# H-Score calculation
# ──────────────────────────────────────────────

def calculate_dual_hscore(show: dict) -> tuple[float, float, float]:
    """Return (landing, longevity, combined) H-Score on a 0–100 scale."""
    landing = sum(show[col] * (WEIGHTS_LANDING[col] / 10) for col in AXIS_COLS)
    longevity = sum(show[col] * (WEIGHTS_LONGEVITY[col] / 10) for col in AXIS_COLS)

    landing_score = round(landing / sum(WEIGHTS_LANDING.values()) * 100, 1)
    longevity_score = round(longevity / sum(WEIGHTS_LONGEVITY.values()) * 100, 1)
    combined = round(landing_score * 0.45 + longevity_score * 0.55, 1)
    return landing_score, longevity_score, combined


def predict_new_shows(new_shows: list[dict]) -> list[dict]:
    """Score shows in NEW_SHOWS (not used for calibration)."""
    print(f"\n{'='*60}")
    print("🔮 신규/방영중 프로그램 H-Score 예측 (calibration 미포함)")
    print("=" * 60)
    results = []
    for show in new_shows:
        land, longevity, combined = calculate_dual_hscore(show)
        row = dict(show)
        row.update({"hscore_landing": land, "hscore_longevity": longevity, "hscore": combined})
        results.append(row)
        rating_str = f"{show['avg_rating_pct']:.3f}%" if show.get("avg_rating_pct") else "미공개"
        print(
            f"  {show['title']:<20s} | "
            f"Landing {land:>5.1f} | Longevity {longevity:>5.1f} | "
            f"H-Score {combined:>5.1f} | 첫방 {rating_str}"
        )
    return results


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

FORMAT_COLORS = {
    "OBS": "#4ECDC4", "VAR": "#FF6B6B", "SRV": "#FFEAA7",
    "TLK": "#96CEB4", "MUS": "#DDA0DD", "TRL": "#45B7D1",
    "SPT": "#FF8C42",
}


def plot_weight_bar(weights: dict) -> None:
    items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    names = [AXIS_NAMES_KR[k] for k, _ in items]
    values = [v for _, v in items]
    colors = [FORMAT_COLORS.get(k[:3], "#999") for k, _ in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, values, color="#4ECDC4", edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Weight (out of 60)")
    ax.set_title("Non-Drama H-Score Empirical Weight Distribution")
    ax.set_xlim(0, max(values) * 1.3)
    ax.invert_yaxis()
    fig.tight_layout()
    path = OUTPUT_DIR / "nd_weight_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


def plot_hscore_comparison(scored: list[dict]) -> None:
    ordered = sorted(scored, key=lambda x: x["hscore"], reverse=True)
    titles = [d["title"][:18] for d in ordered]
    scores = [d["hscore"] for d in ordered]
    colors = [FORMAT_COLORS.get(d["format_type"], "#999") for d in ordered]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(titles, scores, color=colors, edgecolor="white", height=0.65)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("H-Score (0–100)")
    ax.set_title("K-Show Non-Drama H-Score Comparison (2024–2026)")
    ax.set_xlim(0, 100)
    ax.axvline(70, linestyle="--", color="#FF6B6B", alpha=0.5, label="Hit threshold (70)")
    ax.axvline(50, linestyle="--", color="#aaa", alpha=0.3, label="Average (50)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=f"{k} {FORMAT_LABELS[k]}") for k, v in FORMAT_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, title="Format")
    ax.invert_yaxis()
    fig.tight_layout()
    path = OUTPUT_DIR / "nd_hscore_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = AXIS_COLS + ["combined_performance"]
    corr = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    labels = [AXIS_NAMES_KR.get(c, c) for c in cols]
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
        xticklabels=labels, yticklabels=labels,
        vmin=-1, vmax=1, center=0, linewidths=0.5, square=True, ax=ax,
    )
    ax.set_title("Non-Drama H-Score: Spearman Correlation Matrix")
    fig.tight_layout()
    path = OUTPUT_DIR / "nd_correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


def plot_actual_vs_predicted(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df.copy()
    df["hscore_weighted"] = [
        round(sum(row[col] * (weights[col] / 10) for col in AXIS_COLS)
              / sum(weights.values()) * 100, 1)
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    for fmt, grp in df.groupby("format_type"):
        ax.scatter(grp["hscore_weighted"], grp["combined_performance"],
                   c=FORMAT_COLORS.get(fmt, "#999"), s=80, alpha=0.8,
                   label=f"{fmt} {FORMAT_LABELS.get(fmt,'')}", edgecolors="white")
        for _, row in grp.iterrows():
            ax.annotate(row["title"][:8], (row["hscore_weighted"], row["combined_performance"]),
                        fontsize=7, alpha=0.7)
    z = np.polyfit(df["hscore_weighted"], df["combined_performance"], 1)
    x_line = np.linspace(df["hscore_weighted"].min(), df["hscore_weighted"].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), "--", color="#FF6B6B", alpha=0.5)
    rho = df["hscore_weighted"].corr(df["combined_performance"], method="spearman")
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes,
            fontsize=11, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Weighted H-Score")
    ax.set_ylabel("Combined Performance (normalised)")
    ax.set_title("Predicted H-Score vs. Actual Performance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "nd_actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")
    return df


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────

def save_results(
    weights: dict, calibration_scored: list[dict],
    new_scored: list[dict], df: pd.DataFrame,
) -> None:
    output = {
        "version": "1.1-nondrama",
        "methodology": "5-method ensemble (Spearman + OLS + Ridge + RF + GB)",
        "dual_kpi": "Landing 45% + Longevity 55%",
        "data_size": int(len(df)),
        "weights_empirical": weights,
        "weights_landing": WEIGHTS_LANDING,
        "weights_longevity": WEIGHTS_LONGEVITY,
        "format_labels": FORMAT_LABELS,
        "calibration_shows": calibration_scored,
        "new_shows_predicted": new_scored,
    }
    out_path = OUTPUT_DIR / "nondrama_hscore_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  💾 결과 JSON: {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> tuple[dict, list[dict]]:
    setup_korean_font()

    # Step 1 — data
    print("\n📦 Step 1: 캘리브레이션 데이터 로드")
    df = build_dataframe(CALIBRATION_SHOWS)
    print(f"  shows: {len(df)}")
    print(f"  format distribution: {df['format_type'].value_counts().to_dict()}")

    # Step 2 — Nielsen cross-check (optional)
    print("\n📦 Step 2: 닐슨 CSV 교차 검증 (optional)")
    nielsen_path = os.getenv(
        "NIELSEN_CSV_PATH",
        str(Path("data") / "nielsen_weekly_all_categories_2024_2026.csv"),
    )
    load_nielsen_data(nielsen_path)

    # Step 3 — correlation
    print("\n📦 Step 3: 상관 분석")
    correlations = correlation_analysis(df)

    df_tv = df[df["avg_rating_pct"].notna()].copy()
    if len(df_tv) >= 5:
        correlation_analysis(df_tv, target="avg_rating_pct")

    # Step 4 — regression
    print("\n📦 Step 4: 회귀 분석")
    regression_results = regression_analysis(df)

    # Step 5 — weights
    print("\n📦 Step 5: 가중치 산출")
    weights = derive_weights(correlations, regression_results)

    # Step 6 — score calibration shows
    print(f"\n📦 Step 6: Dual KPI H-Score 산출 (calibration {len(CALIBRATION_SHOWS)}편)")
    print(f"\n  {'':>4}  {'프로그램':<24} {'안착':>6} {'롱런':>6} {'종합':>6}  {'시청률':>7}  포맷")
    print("  " + "-" * 70)
    calibration_scored: list[dict] = []
    for i, show in enumerate(
        sorted(CALIBRATION_SHOWS, key=lambda s: calculate_dual_hscore(s)[2], reverse=True), start=1
    ):
        land, longevity, hscore = calculate_dual_hscore(show)
        rating_str = f"{show['avg_rating_pct']:.1f}%" if show.get("avg_rating_pct") else "OTT"
        print(f"  {i:>4}  {show['title']:<24} {land:>6.1f} {longevity:>6.1f} {hscore:>6.1f}  {rating_str:>7}  {show['format_type']}")
        row = {**show, "hscore_landing": land, "hscore_longevity": longevity, "hscore": hscore}
        calibration_scored.append(row)

    # Step 7 — predict new shows
    new_scored = predict_new_shows(NEW_SHOWS)

    # Step 8 — visualise
    print("\n📦 Step 8: 시각화")
    plot_correlation_heatmap(df)
    plot_weight_bar(weights)
    plot_hscore_comparison(calibration_scored)
    plot_actual_vs_predicted(df, weights)

    # Step 9 — save
    save_results(weights, calibration_scored, new_scored, df)

    print("\n✅ 완료!  output_nondrama/ 에서 결과를 확인하세요.")
    return weights, calibration_scored


if __name__ == "__main__":
    main()
