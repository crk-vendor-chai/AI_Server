#!/usr/bin/env python3
"""Idempotently deploy PNT system dashboards to the external Grafana."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboards"
BASE = os.environ.get("GRAFANA_URL", "https://aipnt.atcrk.co.kr/grafana").rstrip("/")
TOKEN_FILE = Path(os.environ.get("GRAFANA_TOKEN_FILE", ROOT.parent / "secrets" / "grafana_service_account.token"))


def load_token():
    token = TOKEN_FILE.read_text().strip()
    if "=" in token and not token.startswith("glsa_"):
        token = token.split("=", 1)[1].strip()
    if not token:
        raise RuntimeError(f"empty Grafana token: {TOKEN_FILE}")
    return token


TOKEN = load_token()


def request(method, path, payload=None, allowed=(200,)):
    command = [
        "curl", "-sS", "--max-time", "30", "-X", method,
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-w", "\\n%{http_code}",
    ]
    if payload is not None:
        command += ["--data-binary", "@-"]
    command.append(BASE + path)
    encoded = None if payload is None else json.dumps(payload).encode()
    result = subprocess.run(command, input=encoded, capture_output=True, check=True)
    body, status_text = result.stdout.rsplit(b"\n", 1)
    status = int(status_text)
    if status not in allowed:
        raise RuntimeError(f"{method} {path} returned HTTP {status}: {body.decode(errors='replace')[:300]}")
    return status, json.loads(body) if body.strip() else {}


request("GET", "/api/datasources/uid/pnt-prometheus")
request("GET", "/api/datasources/uid/pnt-prometheus/health")
print("verified datasource: PNT System Metrics")

_, folders = request("GET", "/api/search?type=dash-folder")
folder = next((item for item in folders if item.get("title") == "CRK AI"), None)
if folder is None:
    _, folder = request("POST", "/api/folders", {"uid": "crk-ai", "title": "CRK AI"})
folder_uid = folder["uid"]

for path in sorted(DASHBOARD_DIR.glob("*.json")):
    dashboard = json.loads(path.read_text())
    payload = {
        "dashboard": dashboard,
        "folderUid": folder_uid,
        "overwrite": True,
        "message": "Provision PNT system monitoring dashboard",
    }
    _, response = request("POST", "/api/dashboards/db", payload)
    print(f"provisioned dashboard: {dashboard['title']} ({response.get('url', '')})")
