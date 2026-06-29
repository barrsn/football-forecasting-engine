# Football Forecasting Engine - הסבר בעברית

מערכת לחיזוי הסתברויות בכדורגל נבחרות ולדוחות מונדיאל 2026. הדגש המרכזי
בריפו הוא מניעת דליפת מידע: כל פיצ'ר חייב להיות ידוע לפני מועד המשחק.

## מצב נוכחי

המודל המקודם מתועד ב-[reports/MODEL_CARD.md](reports/MODEL_CARD.md).

| פיצול / מדיניות | Log Loss | Brier | RPS | דיוק | כיסוי |
|---|---:|---:|---:|---:|---:|
| Holdout, כל המשחקים | 0.8153 | 0.4787 | 0.1563 | 61.93% | 100.00% |
| Holdout, בחירות בביטחון גבוה | n/a | n/a | n/a | 83.71% | 40.37% |
| Rolling validation, ביטחון גבוה | n/a | n/a | n/a | 78.55% | 38.91% |

מדיניות הביטחון הגבוה היא שכבת hard picks בלבד. כל משחק עדיין מקבל הסתברויות
מלאות לניצחון צד 1, תיקו וניצחון צד 2, וסימולציות טורניר משתמשות בהסתברויות
ולא בתוויות קשיחות.

## מה הריפו כולל

- בדיקת סכימה ונרמול שמות נבחרות
- קליטת מקורות מקומיים עם hashes ו-snapshots משוחזרים
- Elo ו-rolling features שמחושבים רק ממשחקים קודמים
- מודלי baseline, Logistic, Poisson, Dixon-Coles ו-HistGBM
- CatBoost ו-LightGBM כאופציונליים בלבד
- כיול הסתברויות ו-ensembles שנבחרים על validation כרונולוגי
- backtesting כרונולוגי ו-holdout קבוע
- מדדי Log Loss, Brier, RPS, ECE ודיוק
- מדיניות abstain לבחירות בביטחון גבוה
- דוח Streamlit לצפייה בתוצאות המודל ובתחזיות מונדיאל
- בדיקות offline ללא אינטרנט

## הרצה מהירה

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
python scripts/run_sample_pipeline.py
```

הרצת האפליקציה:

```bash
python -m pip install -e ".[dev,app]"
streamlit run app/streamlit_app.py
```

בסביבה המקומית המתועדת:

```powershell
conda run -n trade310 python -m pytest -q
conda run -n trade310 python -m ruff check src tests scripts app
conda run -n trade310 python scripts/run_sample_pipeline.py
```

## זרימת עבודה נכונה

1. בדיקת סכימת נתונים.
2. נרמול שמות קבוצות.
3. מיון כל המשחקים לפי תאריך.
4. בניית ratings לפי as-of-date.
5. בניית rolling features רק ממידע קודם.
6. אימון baseline ומודלים מועמדים.
7. הערכה עם Log Loss, Brier ו-RPS.
8. כיול הסתברויות על validation כרונולוגי.
9. סימולציית טורניר מתוך הסתברויות בלבד.
10. כתיבת הנחות ומגבלות ב-Model Card.

## מבנה הריפו

```text
app/                    אפליקציית Streamlit לדוחות
configs/                קונפיגורציות פרויקט ופיצ'רים
data/sample/            דאטה קטן לבדיקות offline
docs/                   חוזי נתונים, הערכה וסימולציה
models/*.metadata.json  מטאדאטה ציבורית של מודלים
reports/                Model Card ותוצרי הערכה נבחרים
scripts/                פקודות CLI דקות
src/football_forecast/  קוד החבילה
tests/                  בדיקות
```

קבצי raw/interim/processed, מודלים בינאריים וקבצי cache לא אמורים להיכנס ל-Git.
הריפו שומר metadata, checksums ותוצרי דוח נבחרים במקום artefacts כבדים.
