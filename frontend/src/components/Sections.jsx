import { GITHUB_URL } from "../api.js";
import { formatNumber } from "../utils.js";

const MODEL_NAMES = {
  baseline: "Baseline (training mean)",
  linear_regression: "Linear Regression",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
};

export function HowItWorks() {
  return (
    <section className="section" id="how-it-works" aria-labelledby="how-heading">
      <h2 id="how-heading">How the prediction works</h2>
      <ol className="steps">
        <li>
          <strong>You enter property details.</strong> The form is built from the fields the trained model supports, and
          the server checks every value before using it.
        </li>
        <li>
          <strong>The saved model runs once.</strong> The same preprocessing used in training (filling gaps, encoding
          locations, scaling numbers) is stored inside the model file, so nothing is retrained when you press the button.
        </li>
        <li>
          <strong>You get an estimate, not a valuation.</strong> The result shows the model&apos;s measured typical error
          from held-out test data so you can judge how much to trust it.
        </li>
      </ol>
    </section>
  );
}

export function ModelMetrics({ model, status, message }) {
  return (
    <section className="section" id="model" aria-labelledby="model-heading">
      <h2 id="model-heading">Model evaluation</h2>
      {status === "loading" && <p className="muted">Loading results…</p>}
      {status === "error" && (
        <p className="banner banner-error" role="alert">
          Evaluation results are unavailable: {message}
        </p>
      )}
      {status === "ready" && (
        <>
          <p>
            All models were evaluated on {formatNumber(model.rows.rows_test)} held-out listings they never trained on.
            Errors are in lakh rupees (1 lakh = ₹1,00,000). Lower MAE and RMSE are better; R² closer to 1 is better. CV RMSE is the 5-fold cross-validated error on training data, used to pick the model.
          </p>
          <div className="table-wrap" tabIndex={0} role="region" aria-label="Model comparison table">
            <table>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">MAE (lakh)</th>
                  <th scope="col">RMSE (lakh)</th>
                  <th scope="col">R²</th>
                  <th scope="col">CV RMSE (lakh)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(model.metrics).map(([name, m]) => (
                  <tr key={name} className={name === model.selected ? "selected" : undefined}>
                    <th scope="row">
                      {MODEL_NAMES[name] || name}
                      {name === model.selected && <span className="badge">in use</span>}
                    </th>
                    <td>{m.test.mae.toFixed(2)}</td>
                    <td>{m.test.rmse.toFixed(2)}</td>
                    <td>{m.test.r2.toFixed(3)}</td>
                    <td>{m.cv ? m.cv.rmse_mean.toFixed(2) : "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">
            Trained {new Date(model.trained_at).toLocaleDateString("en-IN", { dateStyle: "medium" })}. The model in use was
            chosen by lowest cross-validated RMSE on the training data, so the test figures are a final check.
            {model.baseline_beaten === false && " It did not beat the baseline, so treat its output with great caution."}
          </p>
        </>
      )}
    </section>
  );
}

export function About({ dataset, versions }) {
  return (
    <section className="section" id="about" aria-labelledby="about-heading">
      <h2 id="about-heading">Dataset and technology</h2>
      <div className="two-col">
        <div>
          <h3>Dataset</h3>
          {dataset ? (
            <>
              <p>
                <a href={dataset.source_url} target="_blank" rel="noreferrer">
                  {dataset.name}
                </a>{" "}
                covering {dataset.coverage}. Prices are in {dataset.price_unit}.
              </p>
              <p className="muted">{dataset.license_note}</p>
            </>
          ) : (
            <p className="muted">Dataset details appear once the backend has a trained model.</p>
          )}
        </div>
        <div>
          <h3>Built with</h3>
          <ul className="plain-list">
            <li>Python, pandas, NumPy, scikit-learn, joblib</li>
            <li>Flask REST API</li>
            <li>React with Vite</li>
          </ul>
          {versions?.["scikit-learn"] && <p className="muted">Trained with scikit-learn {versions["scikit-learn"]}.</p>}
        </div>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="footer">
      <p>House Price Predictor. Estimates only, not financial or valuation advice.</p>
      <p>
        <a href={GITHUB_URL} target="_blank" rel="noreferrer">
          View the source on GitHub
        </a>
      </p>
    </footer>
  );
}
