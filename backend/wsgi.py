"""Entry point: `python wsgi.py` for development, or `flask --app wsgi run`."""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
