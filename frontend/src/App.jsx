import { useEffect, useState } from "react";
import { ApiError, fetchMetadata, predictPrice } from "./api.js";
import PredictionForm from "./components/PredictionForm.jsx";
import ResultCard from "./components/ResultCard.jsx";
import { About, Footer, HowItWorks, ModelMetrics } from "./components/Sections.jsx";

export default function App() {
  const [meta, setMeta] = useState({ status: "loading", data: null, message: "" });
  const [result, setResult] = useState(null);
  const [lastValues, setLastValues] = useState({});
  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState("");
  const [serverFieldErrors, setServerFieldErrors] = useState({});

  useEffect(() => {
    let cancelled = false;
    fetchMetadata()
      .then((data) => !cancelled && setMeta({ status: "ready", data, message: "" }))
      .catch((err) => !cancelled && setMeta({ status: "error", data: null, message: err.message }));
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSubmit = async (payload, rawValues) => {
    setLoading(true);
    setServerError("");
    setServerFieldErrors({});
    setLastValues(rawValues);
    try {
      setResult(await predictPrice(payload));
    } catch (err) {
      if (err instanceof ApiError && err.status === 422 && Object.keys(err.fieldErrors).length > 0) {
        setServerFieldErrors(err.fieldErrors);
        setServerError("Please fix the highlighted fields.");
      } else {
        setServerError(err.message || "Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = () => {
    setResult(null);
    setServerError("");
  };

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <header className="site-header">
        <span className="brand">House Price Predictor</span>
        <nav aria-label="Sections">
          <a href="#how-it-works">How it works</a>
          <a href="#model">Evaluation</a>
          <a href="#about">About</a>
        </nav>
      </header>

      <main id="main">
        <section className="hero">
          <div className="hero-copy">
            <h1>What might a Bengaluru home cost?</h1>
            <p>
              Enter a few property details and a machine learning model trained on real listing data returns an estimate in
              rupees, along with how far off its estimates typically are.
            </p>
          </div>

          <div className="card">
            {meta.status === "loading" && <p className="muted">Connecting to the model…</p>}
            {meta.status === "error" && (
              <div className="banner banner-error" role="alert">
                <strong>The predictor is not available right now.</strong>
                <p>{meta.message}</p>
              </div>
            )}
            {meta.status === "ready" && !result && (
              <>
                <h2 className="card-title">Property details</h2>
                <PredictionForm
                  fields={meta.data.fields}
                  initialValues={lastValues}
                  onSubmit={handleSubmit}
                  loading={loading}
                  serverError={serverError}
                  serverFieldErrors={serverFieldErrors}
                />
              </>
            )}
            {meta.status === "ready" && result && <ResultCard result={result} onEdit={handleEdit} />}
          </div>
        </section>

        <HowItWorks />
        <ModelMetrics model={meta.data?.model} status={meta.status} message={meta.message} />
        <About dataset={meta.data?.dataset} versions={meta.data?.library_versions} />
      </main>
      <Footer />
    </>
  );
}
