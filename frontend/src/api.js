const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:5000").replace(/\/$/, "");

export const GITHUB_URL =
  import.meta.env.VITE_GITHUB_URL || "https://github.com/your-username/house-price-predictor";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "", fieldErrors = {} } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.fieldErrors = fieldErrors;
  }
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new ApiError(
      `Cannot reach the backend at ${API_URL}. Start it with "python wsgi.py" in the backend folder.`,
      { code: "network" }
    );
  }
  let body = null;
  try {
    body = await response.json();
  } catch {
    /* non-JSON response */
  }
  if (!response.ok) {
    throw new ApiError(body?.message || `Request failed (${response.status}).`, {
      status: response.status,
      code: body?.error || "",
      fieldErrors: body?.errors || {},
    });
  }
  return body;
}

export const fetchMetadata = () => request("/api/metadata");

export const predictPrice = (values) =>
  request("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  });
