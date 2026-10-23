import { useId, useState } from "react";
import { formatNumber, toPayload, validateForm } from "../utils.js";

export default function PredictionForm({ fields, initialValues, onSubmit, loading, serverError, serverFieldErrors }) {
  const [values, setValues] = useState(initialValues || {});
  const [errors, setErrors] = useState({});
  const uid = useId();

  const shownErrors = { ...serverFieldErrors, ...errors };

  const handleChange = (name, value) => {
    setValues((v) => ({ ...v, [name]: value }));
    if (errors[name]) setErrors((e) => ({ ...e, [name]: undefined }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const found = validateForm(fields, values);
    setErrors(found);
    if (Object.keys(found).length > 0) {
      const first = fields.find((f) => found[f.name]);
      document.getElementById(`${uid}-${first.name}`)?.focus();
      return;
    }
    onSubmit(toPayload(fields, values), values);
  };

  return (
    <form className="form" onSubmit={handleSubmit} noValidate aria-busy={loading}>
      <div className="form-grid">
        {fields.map((f) => {
          const id = `${uid}-${f.name}`;
          const errId = `${id}-error`;
          const error = shownErrors[f.name];
          const common = {
            id,
            name: f.name,
            value: values[f.name] ?? "",
            "aria-invalid": error ? "true" : undefined,
            "aria-describedby": error ? errId : undefined,
            disabled: loading,
            onChange: (e) => handleChange(f.name, e.target.value),
          };
          return (
            <div className={`field${f.name === "location" ? " field-wide" : ""}`} key={f.name}>
              <label htmlFor={id}>{f.label}</label>
              {f.type === "category" && f.name === "location" && (
                <>
                  <input {...common} list={`${id}-options`} autoComplete="off" placeholder="Start typing, e.g. Whitefield" />
                  <datalist id={`${id}-options`}>
                    {f.options.map((o) => (
                      <option key={o} value={o} />
                    ))}
                  </datalist>
                </>
              )}
              {f.type === "category" && f.name !== "location" && (
                <select {...common}>
                  <option value="">Select…</option>
                  {f.options.map((o) => (
                    <option key={o} value={o}>
                      {o.replace(/\s+/g, " ")}
                    </option>
                  ))}
                </select>
              )}
              {f.type === "number" && (
                <input {...common} type="number" inputMode={f.integer ? "numeric" : "decimal"} min={f.min} max={f.max} step={f.integer ? 1 : "any"} />
              )}
              {f.type === "number" && !error && (
                <span className="hint">
                  {formatNumber(f.min)} to {formatNumber(f.max)}
                </span>
              )}
              {error && (
                <span className="error" id={errId} role="alert">
                  {error}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {serverError && (
        <p className="banner banner-error" role="alert">
          {serverError}
        </p>
      )}

      <button className="button" type="submit" disabled={loading}>
        {loading ? "Estimating…" : "Estimate price"}
      </button>
    </form>
  );
}
