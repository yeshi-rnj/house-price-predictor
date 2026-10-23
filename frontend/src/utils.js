const number = new Intl.NumberFormat("en-IN");
export const formatNumber = (n) => number.format(n);

/** Client-side checks mirror the server rules; the server remains the authority. */
export function validateForm(fields, values) {
  const errors = {};
  for (const f of fields) {
    const raw = values[f.name];
    const empty = raw === undefined || raw === null || String(raw).trim() === "";
    if (empty) {
      errors[f.name] = `${f.label} is required.`;
      continue;
    }
    if (f.type === "category") {
      const known = f.options.some((o) => o.toLowerCase() === String(raw).trim().toLowerCase());
      if (!known) errors[f.name] = `Choose a ${f.label.toLowerCase()} from the list.`;
      continue;
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) errors[f.name] = `${f.label} must be a number.`;
    else if (f.integer && !Number.isInteger(n)) errors[f.name] = `${f.label} must be a whole number.`;
    else if (n < f.min || n > f.max)
      errors[f.name] = `${f.label} must be between ${formatNumber(f.min)} and ${formatNumber(f.max)}.`;
  }
  return errors;
}

/** Convert form strings to the types the API expects. */
export function toPayload(fields, values) {
  const payload = {};
  for (const f of fields) {
    payload[f.name] = f.type === "number" ? Number(values[f.name]) : String(values[f.name]).trim();
  }
  return payload;
}
