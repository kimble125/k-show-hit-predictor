# data/

이 폴더에는 캘리브레이션에 사용하는 외부 데이터 파일을 넣습니다.

## Nielsen Weekly Ratings CSV (optional)

`nielsen_weekly_all_categories_2024_2026.csv`

닐슨코리아 주간 시청률 데이터입니다. 스크립트의 Step 2 교차 검증에 사용됩니다.
이 파일이 없어도 스크립트는 내장 캘리브레이션 데이터만으로 실행됩니다.

### Expected columns

| Column | Type | Description |
|---|---|---|
| `program` | str | 프로그램명 |
| `metric_type` | str | `household_rating` 또는 `audience_count` |
| `metric_value` | float | 시청률(%) 또는 시청자수(천명) |
| `begin_date` | str | 주차 시작일 (YYYY-MM-DD) |
| `category` | str | `지상파` / `종편` / `케이블` |
| `area` | str | 지역 코드 |

### How to specify the path

기본 경로:
```
data/nielsen_weekly_all_categories_2024_2026.csv
```

환경변수로 직접 지정:
```bash
export NIELSEN_CSV_PATH="/path/to/your/nielsen.csv"
python non_drama_hscore_calibration.py
```

또는 `scripts/add_nielsen_csv_and_push.sh` 스크립트를 사용하면
Google Drive에서 자동으로 복사 후 커밋할 수 있습니다.

## Notes

- 닐슨 CSV는 저작권 문제로 이 저장소에 직접 포함되지 않습니다.
- OTT 전용 프로그램(Netflix, TVING, 쿠팡플레이 등)은 시청률 대신
  `brand_reputation` 지수(한국기업평판연구소)를 성과 지표로 사용합니다.
