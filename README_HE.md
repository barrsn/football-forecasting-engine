# Football Forecasting Engine - הסבר בעברית

מערכת לחיזוי הסתברויות בכדורגל נבחרות ולסימולציית מונדיאל 2026, עם דגש על
מניעת דליפת מידע והערכה כרונולוגית.

## מה כבר ממומש

- בדיקת סכימה קשיחה ונרמול שמות נבחרות
- קליטת קבצים מקומיים עם commit ו-SHA-256
- snapshots לפי זמן חיתוך
- Elo ו-rolling features שאינם משתמשים במשחק הנוכחי או במשחק באותו זמן
- baselines, Poisson, Dixon-Coles, Logistic ו-HistGradientBoosting
- calibration, ensemble וקריטריונים לקידום מודל
- backtesting כרונולוגי
- סימולציית 48 נבחרות, הארכה, פנדלים והסתברות הגעה לכל שלב
- בדיקות offline לסכימה, leakage, מודלים וסימולציה

## תוצאות מודל אמיתי

הנוטבוק `notebooks/world_cup_2026_real_models.ipynb` רץ על 32,252 משחקים
שהושלמו עד 10 ביוני 2026. ב-holdout של 1,308 משחקים משנת 2025 ועד החיתוך:

- Log Loss: `0.8373`
- Brier: `0.4914`
- RPS: `0.1620`
- Accuracy: `61.0%`
- Calibration error: `0.0301`

המודל הנבחר הוא Logistic Regression עם משקולות זמן של 8 שנים. Ensemble לא
קודם משום שהשיפור שלו ב-validation היה קטן מסף 1%.

עדיין נדרשים אימות fixtures רשמי, דירוגי FIFA היסטוריים וטבלת Annex C לפני
פרסום תחזית טורניר מלאה.

## הרצה בסביבה trade310

```powershell
conda run -n trade310 python -m pytest -q
conda run -n trade310 ruff check src tests scripts app
conda run -n trade310 python scripts/run_sample_pipeline.py
```

פירוט מלא נמצא בתיקיית `docs/` ובקובץ `reports/MODEL_CARD.md`.
