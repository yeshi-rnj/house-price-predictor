"""Flask application factory."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .config import DEFAULT_MODEL_DIR
from .model_store import ModelStore
from .routes import api

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def create_app(model_dir: Path | str | None = None, cors_origins: str | None = None) -> Flask:
    logging.basicConfig(level=logging.INFO)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # prediction requests are tiny

    model_dir = model_dir or os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR
    app.extensions["model_store"] = ModelStore(model_dir)

    origins = cors_origins or os.environ.get("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    allowed = {o.strip() for o in origins.split(",") if o.strip()}

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        if origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Vary"] = "Origin"
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return jsonify({"error": exc.name.lower().replace(" ", "_"), "message": exc.description}), exc.code

    @app.errorhandler(Exception)
    def unexpected_error(exc: Exception):
        app.logger.exception("Unhandled error")
        return jsonify({"error": "internal_error", "message": "Something went wrong on the server."}), 500

    app.register_blueprint(api)
    return app
