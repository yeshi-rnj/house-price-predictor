import { formatNumber } from "../utils.js";

const LABELS = {
  location: "Location",
  area_type: "Area measured as",
  total_sqft: "Area",
  bhk: "Bedrooms",
  bath: "Bathrooms",
  balcony: "Balconies",
};

function display(name, value) {
  if (name === "total_sqft") return `${formatNumber(value)} sq ft`;
  if (name === "area_type") return String(value).replace(/\s+/g, " ");
  return String(value);
}

export default function ResultCard({ result, onEdit }) {
  const { explanation } = result;
  return (
    <section className="result" aria-labelledby="result-heading" aria-live="polite">
      <h2 id="result-heading" className="visually-hidden">
        Estimated price
      </h2>

      <p className="price">{result.formatted_price}</p>
      <div className="dimension" aria-hidden="true">
        <span>{formatNumber(result.inputs.total_sqft)} sq ft</span>
      </div>
      <p className="price-words">
        About {result.price_in_words}, or roughly ₹{formatNumber(result.price_per_sqft)} per sq ft
      </p>

      {result.warnings.length > 0 && (
        <ul className="banner banner-warn" aria-label="Warnings">
          {result.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      <dl className="summary">
        {Object.keys(LABELS).map((name) => (
          <div key={name}>
            <dt>{LABELS[name]}</dt>
            <dd>{display(name, result.inputs[name])}</dd>
          </div>
        ))}
      </dl>

      <div className="explain">
        <h3>How this estimate was made</h3>
        <p>{explanation.summary}</p>
        {explanation.most_influential_inputs.length > 0 && (
          <p>
            Inputs the model relied on most: <strong>{explanation.most_influential_inputs.join(", ")}</strong>.{" "}
            <span className="muted">{explanation.influence_note}</span>
          </p>
        )}
      </div>

      <p className="disclaimer">{result.disclaimer}</p>

      <button className="button button-secondary" type="button" onClick={onEdit}>
        Edit details
      </button>
    </section>
  );
}
