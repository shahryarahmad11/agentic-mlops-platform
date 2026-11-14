"""MLOps agent package."""

import os

# MLflow prints an advisory hint on import; keep CLI and logs clean.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
