# House Price Predictor

A full-stack machine learning web app that estimates Bengaluru house prices in Indian rupees (₹).
Users enter location, area, bedrooms, bathrooms, balconies and area type; a trained scikit-learn
pipeline returns an estimate, the model's measured typical error, and a disclaimer.

> **Status:** code complete, but **no model has been trained yet** and **no screenshots exist yet**.
> The Model evaluation section of the site stays empty until you train on the real dataset.
> Add screenshots and your real metrics to this README only after you have run the app.

## What has and has not been verified

Checked in the build environment (Python 3.12, scikit-learn 1.8, pandas 3.0, Flask 3.1):
- 23 backend unit/API tests pass (`python -m unittest`). They use a tiny model trained on **synthetic**
  data in a temporary folder, only to exercise the code.
- `explore.py` and `train.py` ran end to end on throwaway fake data, including a realistic-scale run
  (about 12,000 rows, 300 locations) to measure training time (about 2.5 minutes) and artifact size
  (the largest candidate, Random Forest, was about 14.5 MB compressed; well under GitHub's 100 MB limit).
  Accuracy from those runs is meaningless because the data was synthetic.
- The React app bundled and rendered in headless Chromium with a stubbed API and no console errors.

**Not verified** (the build environment had no internet):
- Training on the real Kaggle file. Column names and formats below are from the dataset's usual layout
  and **must be confirmed** with `python explore.py` (it fails with a clear message if columns differ).
- `npm install`, `npm run dev`, `npm run build` with real Vite.
- Docker build, `render.yaml` deploy, and the GitHub Actions workflow (written against current docs, never run).
- Python 3.11 specifically, and Windows PowerShell commands.
- The dataset URL and license. Open the Kaggle page and read the license before publishing anything.

## Project structure

```
house-price-predictor/
  backend/
    app/            Flask app: config, features (cleaning), model_store, validation, predictor, routes
    tests/          unittest-based tests (pytest also discovers them)
    models/         model.joblib, metadata.json, metrics.json, model-requirements.txt (from train.py)
    data/raw/       put the downloaded CSV here (ignored by Git)
    data/processed/ reserved for derived data (ignored by Git)
    explore.py      exploratory analysis and leakage notes
    train.py        reproducible training and model comparison
    wsgi.py         entry point (dev server, or gunicorn wsgi:app in production)
    Dockerfile      container image for the API
  frontend/         React + Vite app
  render.yaml       Render deployment blueprint
  .github/workflows CI (tests and frontend build)
```

Deviation from a flat layout: `data/` is split into `raw/` (never edited) and `processed/`.

## Dataset

- Name: Bengaluru House Price Data (Kaggle)
- URL: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data
- License: **check the Kaggle page**. The CSV is not included in this repo; each user downloads it.
- Target: `price`, in lakh rupees (1 lakh = ₹1,00,000).
- Expected columns: `area_type, availability, location, size, society, total_sqft, bath, balcony, price`.

Features used: `location`, `area_type`, `total_sqft`, `bhk` (parsed from `size`), `bath`, `balcony`.
Excluded on purpose: `society` (mostly missing, near-unique labels), `availability` (dates), and any
price-per-sq-ft column (derived from the target, so it would leak it).

### Cleaning decisions (see `app/features.py`; every dropped row is counted)
- Exact duplicate rows removed.
- `total_sqft` ranges like "2100 - 2850" become the midpoint; values with units (e.g. "Sq. Meter") are dropped, not guessed.
- `size` ("2 BHK", "4 Bedroom") parsed to a number of bedrooms.
- Rows outside sane limits are dropped (area 200-30,000 sq ft, 1-10 bedrooms/bathrooms, 0-5 balconies).
- Heuristics, to revisit after you read `explore.py` output: at least 300 sq ft per bedroom, at most bedrooms + 2 bathrooms.
- Missing `balcony` is kept and filled with the training median inside the pipeline.
- Extreme price-per-sq-ft rows (outside the 1st-99th percentile) are trimmed from the **training split only**.
  The test split is never trimmed, so test metrics reflect realistic data.

## Setup (Windows PowerShell)

Prerequisites: Python 3.11 (3.12 should also work), Node.js 20.19+ or 22.12+, Git.

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

### 1. Get the data
Download the CSV from the Kaggle page (free account required) and save it as
`backend/data/raw/Bengaluru_House_Data.csv`.

### 2. Explore, then train
```powershell
python explore.py
python train.py
```
`train.py` prints dropped-row counts, then for four candidates (baseline, Linear Regression, Random
Forest, Gradient Boosting on a log-price target) the 5-fold cross-validated RMSE on the training split
and the test MAE/RMSE/R². Output files appear in `backend/models/`. Random seed: 42.

Reading the metrics (all in lakh rupees): **MAE** is the average size of the error; **RMSE** is similar
but punishes large errors more; **R²** is the share of price variation explained (1 is perfect, 0 is no
better than predicting the average).

**Model selection:** the winner is the non-baseline model with the lowest cross-validated RMSE on the
training split. The test split is scored once, afterwards, so it is an honest final check rather than
part of the selection. The saved model is fitted on the training split only (not refit on all data), so
the reported numbers describe the exact file being served. If the selected model does not beat the
baseline, `train.py` prints a warning and the site shows it. Do not deploy such a model.

**Decide whether the model is good enough to deploy.** Read the test R² and MAE against typical prices
before publishing. The numbers are measured, not promised, and I have not seen them for the real data.

`train.py` also writes `models/model-requirements.txt`, the exact library versions used for training.
Deploy builds install those pins, so the server and the saved model always agree. The API's
`/api/health` reports a warning if the versions ever differ.

### 3. Run the tests
```powershell
python -m unittest discover -s tests -t . -v
# or: pytest
```

### 4. Run the API
```powershell
python wsgi.py
```
```powershell
curl.exe http://localhost:5000/api/health
curl.exe http://localhost:5000/api/metadata
curl.exe -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" `
  -d '{\"location\":\"<a location from /api/metadata>\",\"area_type\":\"<an area type from /api/metadata>\",\"total_sqft\":1200,\"bhk\":2,\"bath\":2,\"balcony\":1}'
```
Use values taken from `/api/metadata`, because location names come from the dataset.

### 5. Run the frontend (second terminal)
```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```
Open http://localhost:5173. If `npm install` complains about versions, scaffold a fresh project with
`npm create vite@latest` (React template) and copy this repo's `src/` and `index.html` over it.

Restarting the backend does **not** retrain: the saved pipeline is loaded from `backend/models/`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | 200 when running with a model loaded, 503 otherwise |
| GET | `/api/metadata` | form fields, allowed categories, saved metrics, dataset info |
| POST | `/api/predict` | validates input, returns a price estimate |

Success response shape (placeholder values, not a real prediction):
```json
{ "predicted_price": 7500000, "currency": "INR", "formatted_price": "₹75,00,000",
  "price_in_words": "₹75 lakh", "price_per_sqft": 6250, "inputs": {}, "explanation": {},
  "warnings": [], "disclaimer": "..." }
```
Errors: 400 invalid JSON, 422 validation failures (`errors` maps field to message), 503 model missing.
Predictions are rounded to the nearest ₹1,000. The "explanation" contains only measured facts: model
name, training/test sizes, measured test MAE, and permutation importance from the test set.

## Security notes
- No secrets are needed. `.env` files are ignored by Git; only `.env.example` files are committed.
- Inputs are validated, size-limited (16 KB) and never evaluated as code.
- CORS allows only the origins in `CORS_ORIGINS`.
- Only load `.joblib` files you trained yourself; they can execute code when loaded.
- Flask's built-in server is for development only; production uses gunicorn.

## Limitations
- Covers Bengaluru only; one dataset snapshot of unknown date. Prices go stale.
- Listing (asking) prices, not final sale prices.
- Few features: no floor, age, amenities, builder or exact coordinates.
- Rare locations are pooled; estimates there are less reliable.
- Individual errors can be far larger than the average error.
- Trained only on the cleaned ranges above; inputs outside them trigger warnings.

## Ideas for improvement
Cross-validation and hyperparameter search; log-transformed target; gradient boosting; prediction
intervals; location-level price statistics; a model card; Docker; CI running the tests.

## Deployment

The backend and the frontend deploy separately. Free-tier limits (sleep on idle, memory, build time)
change often, so check each platform's current docs and pricing before relying on them.

**Step 0: produce and commit the model** (the dataset itself is never committed):
```powershell
cd backend
python train.py
git add models/model.joblib models/metadata.json models/metrics.json models/model-requirements.txt
```
Train with the same Python minor version the server will use (3.11 here). Check the file size
(`train.py` prints it); GitHub rejects files over 100 MB.

**Backend on Render (Blueprint).** `render.yaml` at the repo root defines the service: root directory
`backend`, build `pip install -r requirements-deploy.txt -r models/model-requirements.txt`, start
`gunicorn ... wsgi:app`, health check `/api/health`. Create it with New > Blueprint in the Render
dashboard, then set `CORS_ORIGINS` to your frontend URL (no trailing slash). Render supplies `PORT`.
Free instances may sleep when idle, so the first request after a pause can be slow.

**Backend with Docker (any container host):**
```powershell
docker build -t house-price-api backend
docker run -p 5000:5000 -e CORS_ORIGINS=http://localhost:5173 house-price-api
```

**Frontend on Vercel or Netlify:** set the project root to `frontend`, build command `npm run build`,
output directory `dist`, and environment variables `VITE_API_URL` (your deployed API URL) and
`VITE_GITHUB_URL`. Vite bakes these in at build time, so redeploy after changing them.

**After deploying:** open `<api-url>/api/health` (expect `"model_loaded": true`), then make a real
prediction from the deployed site and check the browser console for CORS errors.

**CI:** `.github/workflows/ci.yml` runs the backend tests and a frontend build on every push.

## GitHub publishing checklist
- [ ] `git status` shows no `.venv`, `node_modules`, `.env` or dataset files
- [ ] Replace `<Your Name>` in LICENSE and the GitHub URL in `frontend/.env.example`
- [ ] Verify the dataset license and keep the attribution above
- [ ] Train on the real data, paste real metrics here, add screenshots of the running app
- [ ] Run the tests and note the result
- [ ] Commit `backend/models/` artifacts (including `model-requirements.txt`) before deploying
- [ ] Confirm the selected model beats the baseline and you are comfortable with its test error
