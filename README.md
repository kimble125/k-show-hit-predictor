# K-Show Hit Predictor: Non-Drama H-Score Calibration

K-예능/쇼 프로그램의 흥행 가능성을 예측하기 위한 **Non-Drama H-Score 6축 가중치 캘리브레이션** 프로젝트입니다.

이 저장소는 드라마 중심의 흥행 예측 지표를 예능·쇼·리얼리티·서바이벌 등 **비드라마 콘텐츠**에 맞게 재설계하고, 2024–2026년 K-예능/쇼 프로그램 사례를 바탕으로 각 축의 기여도를 분석합니다.

> 핵심 목표: “예능 콘텐츠의 초기 안착력과 롱런 지속력을 데이터 기반으로 어떻게 점수화할 수 있는가?”

---

## 1. Project Overview

`non_drama_hscore_calibration.py`는 다음 작업을 수행합니다.

1. 2024–2026년 K-예능/쇼 프로그램 캘리브레이션 데이터 구성
2. Non-Drama H-Score 6축 점수 계산
3. 닐슨 시청률 및 브랜드평판지수 기반 통합 성과 지표 생성
4. 상관분석, 선형회귀, Ridge, Random Forest, Gradient Boosting 기반 가중치 추정
5. 초기 안착력(Landing)과 롱런 지속력(Longevity)을 분리한 Dual KPI H-Score 계산
6. 분석 결과 차트 및 JSON 저장

---

## 2. Non-Drama H-Score 6 Axes

이 프로젝트는 비드라마 콘텐츠의 흥행 요소를 다음 6축으로 정의합니다.

| Axis | Korean | Description |
|---|---|---|
| `cast_chemistry` | 출연진 케미 파워 | 출연진 인지도, 예능 검증도, 조합의 화학작용 |
| `creator_power` | PD/크리에이터 파워 | PD 전작 성과, 포맷 개발 이력, 연출 신뢰도 |
| `format_power` | 포맷 파워 | 포맷 유형, 시즌 이력, OSMU 가능성, 신선도 |
| `platform_scheduling` | 플랫폼·편성 전략 | 채널 도달력, OTT 동시공개, 시간대 경쟁력 |
| `pre_buzz` | 사전 화제성 | 출연진 발표 반응, 티저 조회수, 기사량, SNS 반응 |
| `concept_trend_fit` | 콘셉트·트렌드 적합도 | 사회 트렌드와의 공명, 새로움과 익숙함의 균형 |

---

## 3. Dual KPI Structure

예능/쇼 콘텐츠는 첫 방송 성과만으로 판단하기 어렵기 때문에, 이 프로젝트는 두 가지 KPI를 분리합니다.

### Landing Score

초기 안착력입니다.

- 첫 3–4회 안에 시청자가 정착하는가?
- 사전 화제성, 출연진 케미, 편성 전략의 영향이 큽니다.

### Longevity Score

롱런 지속력입니다.

- 시즌2 이상 확장 가능한가?
- PD/크리에이터 파워, 포맷 파워, 트렌드 적합도의 영향이 큽니다.

최종 H-Score는 다음 구조를 사용합니다.

```text
Combined H-Score = Landing 45% + Longevity 55%
```

---

## 4. Data

### Built-in calibration data

현재 스크립트에는 2024–2026년 K-예능/쇼 프로그램 캘리브레이션 데이터가 Python list 형태로 포함되어 있습니다.

예시 프로그램:

- 흑백요리사: 요리 계급 전쟁
- 나 혼자 산다
- 미운 우리 새끼
- 최강야구 시즌3
- 삼시세끼
- 유 퀴즈 온 더 블럭
- 피지컬: 100 시즌2
- SNL코리아 시즌5
- 환승연애3
- 서진이네2

### Optional Nielsen CSV

스크립트는 닐슨 주간 시청률 CSV가 있으면 교차 검증용으로 읽습니다.

기본 경로:

```text
data/nielsen_weekly_all_categories_2024_2026.csv
```

또는 환경변수로 직접 지정할 수 있습니다.

```bash
export NIELSEN_CSV_PATH="/path/to/nielsen_weekly_all_categories_2024_2026.csv"
python non_drama_hscore_calibration.py
```

> Note: 원본 CSV가 없는 경우에도 스크립트는 내장 캘리브레이션 데이터만으로 실행됩니다.

---

## 5. Methodology

가중치 산정에는 다음 방법을 함께 사용합니다.

- Spearman correlation
- Pearson correlation
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

최종 가중치는 여러 분석 결과를 정규화한 뒤 앙상블 방식으로 통합합니다.

---

## 6. Output

실행 후 `output_nondrama/` 폴더에 다음 결과물이 생성됩니다.

```text
output_nondrama/
├── nd_correlation_heatmap.png
├── nd_weight_distribution.png
├── nd_hscore_comparison.png
├── nd_actual_vs_predicted.png
└── nondrama_hscore_results.json
```

---

## 7. Installation

```bash
git clone https://github.com/kimble125/k-show-hit-predictor.git
cd k-show-hit-predictor

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 8. Usage

기본 실행:

```bash
python non_drama_hscore_calibration.py
```

CSV 경로를 직접 지정해서 실행:

```bash
NIELSEN_CSV_PATH="data/nielsen_weekly_all_categories_2024_2026.csv" python non_drama_hscore_calibration.py
```

---

## 9. Repository Structure

```text
.
├── non_drama_hscore_calibration.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data/
│   └── README.md
└── output_nondrama/
    └── .gitkeep
```

---

## 10. Limitations

이 프로젝트는 포트폴리오 및 분석 실험용 프로젝트입니다.

- 일부 브랜드평판지수와 OTT 성과는 추정치(`[EST]`)를 포함합니다.
- 캘리브레이션 데이터 수가 제한적이므로 통계적 일반화에는 주의가 필요합니다.
- 6축 점수는 정량 데이터와 도메인 판단이 결합된 휴리스틱 점수입니다.
- 실제 상업적 예측 모델로 사용하려면 더 많은 프로그램, 기간별 성과, SNS/검색량/OTT 순위 데이터가 필요합니다.

---

## 11. Future Improvements

- 캘리브레이션 데이터를 `.csv` 또는 `.json`으로 분리
- 신규 프로그램 입력용 CLI 추가
- 네이버 검색량, YouTube 티저 조회수, SNS 언급량 연동
- 포맷별 별도 가중치 모델 학습
- Streamlit 또는 Gradio 기반 대시보드 제작
- H-Score 산출 결과를 콘텐츠 기획/편성 의사결정 리포트로 자동 변환

---

## 12. Recommended GitHub Metadata

### Description

```text
Non-Drama H-Score calibration for Korean variety and entertainment shows.
```

### Topics

```text
python
data-analysis
machine-learning
content-analytics
korean-variety-show
entertainment-analytics
hscore
regression
random-forest
gradient-boosting
matplotlib
scikit-learn
```

---

## Author

- GitHub: [kimble125](https://github.com/kimble125)
- Blog: [forrest125.tistory.com](https://forrest125.tistory.com/)
