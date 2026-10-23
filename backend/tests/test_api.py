"""API tests. Uses a tiny model trained on synthetic data in a temp folder."""
import tempfile
import unittest
from pathlib import Path

from app import create_app
from tests.helpers import build_test_model

VALID = {
    "location": "Alpha Nagar",
    "area_type": "Super built-up  Area",
    "total_sqft": 1200,
    "bhk": 2,
    "bath": 2,
    "balcony": 1,
}


class ApiWithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        build_test_model(cls._tmp.name)
        cls.client = create_app(model_dir=cls._tmp.name).test_client()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def post(self, payload):
        return self.client.post("/api/predict", json=payload)

    def test_health_ok(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "ok", "model_loaded": True})

    def test_saved_artifacts_support_deployment(self):
        meta = self.client.get("/api/metadata").get_json()
        selected = meta["model"]["selected"]
        self.assertIn("cv", meta["model"]["metrics"][selected])
        files = {p.name for p in Path(self._tmp.name).iterdir()}
        self.assertTrue({"model.joblib", "metadata.json", "metrics.json", "model-requirements.txt"} <= files)
        pins = (Path(self._tmp.name) / "model-requirements.txt").read_text()
        self.assertIn("scikit-learn==", pins)

    def test_metadata_describes_fields_and_metrics(self):
        data = self.client.get("/api/metadata").get_json()
        names = [f["name"] for f in data["fields"]]
        self.assertEqual(names, ["location", "area_type", "total_sqft", "bhk", "bath", "balcony"])
        self.assertIn("Alpha Nagar", data["fields"][0]["options"])
        self.assertIn("test", data["model"]["metrics"][data["model"]["selected"]])
        self.assertNotIn("model_dir", str(data))

    def test_valid_prediction_shape(self):
        res = self.post(VALID)
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["currency"], "INR")
        self.assertIsInstance(body["predicted_price"], int)
        self.assertGreater(body["predicted_price"], 0)
        self.assertTrue(body["formatted_price"].startswith("\u20b9"))
        self.assertIn("disclaimer", body)
        self.assertIn("explanation", body)

    def test_location_match_is_case_insensitive(self):
        res = self.post({**VALID, "location": "  alpha nagar "})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["inputs"]["location"], "Alpha Nagar")

    def test_larger_home_costs_more_in_this_test_model(self):
        small = self.post({**VALID, "total_sqft": 800}).get_json()["predicted_price"]
        large = self.post({**VALID, "total_sqft": 2500, "bhk": 4, "bath": 4}).get_json()["predicted_price"]
        self.assertGreater(large, small)

    def test_missing_fields(self):
        res = self.post({"location": "Alpha Nagar"})
        self.assertEqual(res.status_code, 422)
        errors = res.get_json()["errors"]
        for field in ("area_type", "total_sqft", "bhk", "bath", "balcony"):
            self.assertIn(field, errors)

    def test_unsupported_category(self):
        res = self.post({**VALID, "location": "Atlantis"})
        self.assertEqual(res.status_code, 422)
        self.assertIn("location", res.get_json()["errors"])

    def test_wrong_types_and_ranges(self):
        cases = {
            "total_sqft": ["1200", True, -5, 10, 10**7, float("nan")],
            "bhk": [2.5, 0, 50, "two"],
            "balcony": [-1, 9],
        }
        for field, bad_values in cases.items():
            for bad in bad_values:
                with self.subTest(field=field, value=bad):
                    res = self.post({**VALID, field: bad})
                    self.assertEqual(res.status_code, 422)
                    self.assertIn(field, res.get_json()["errors"])

    def test_non_json_and_non_object_bodies(self):
        res = self.client.post("/api/predict", data="not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)
        res = self.client.post("/api/predict", json=[1, 2, 3])
        self.assertEqual(res.status_code, 400)

    def test_implausible_combination_warns(self):
        body = self.post({**VALID, "total_sqft": 600, "bhk": 6, "bath": 6}).get_json()
        self.assertTrue(body["warnings"])

    def test_unknown_route_and_method_return_json(self):
        self.assertEqual(self.client.get("/api/nope").status_code, 404)
        self.assertEqual(self.client.get("/api/predict").status_code, 405)
        self.assertIn("error", self.client.get("/api/nope").get_json())

    def test_cors_allows_only_configured_origin(self):
        ok = self.client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(ok.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")
        bad = self.client.get("/api/health", headers={"Origin": "http://evil.example"})
        self.assertIsNone(bad.headers.get("Access-Control-Allow-Origin"))


class VersionMismatch(unittest.TestCase):
    def test_health_warns_when_training_versions_differ(self):
        import json
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        build_test_model(tmp.name)
        meta_path = Path(tmp.name) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["library_versions"]["scikit-learn"] = "0.0.1"
        meta_path.write_text(json.dumps(meta))
        res = create_app(model_dir=tmp.name).test_client().get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertIn("scikit-learn", " ".join(res.get_json()["warnings"]))


class ApiWithoutModel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.client = create_app(model_dir=Path(self._tmp.name)).test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def test_health_reports_missing_model(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 503)
        self.assertFalse(res.get_json()["model_loaded"])

    def test_metadata_and_predict_return_503(self):
        self.assertEqual(self.client.get("/api/metadata").status_code, 503)
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 503)

    def test_corrupt_model_file_is_handled(self):
        (Path(self._tmp.name) / "model.joblib").write_bytes(b"garbage")
        (Path(self._tmp.name) / "metadata.json").write_text("{}")
        client = create_app(model_dir=self._tmp.name).test_client()
        self.assertEqual(client.get("/api/health").status_code, 503)


if __name__ == "__main__":
    unittest.main()
