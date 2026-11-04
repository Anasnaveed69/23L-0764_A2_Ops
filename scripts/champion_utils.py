"""Shared helper: find the model version that carries an alias (with a tag fallback)."""
from mlflow.tracking import MlflowClient


def version_by_alias(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception:  # noqa: BLE001 — server without alias support: look for the tag instead
        for v in client.search_model_versions(f"name='{name}'"):
            if v.tags.get("alias") == alias:
                return v
        raise
