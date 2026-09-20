"""Fit/load the explicit v3.5 model, with identical offline/live features."""

import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .actions import button_names, validate_action
from .dataset import load_table
from .features import (
    CATEGORICAL_INDICES,
    FEATURE_NAMES,
    SCHEMA,
    checked_ram,
    extract_features,
)


def train(
    table: Path,
    output: Path,
    *,
    device="auto",
    n_estimators=1,
    fit_mode="fit_with_cache",
    seed=0,
    model_path=None,
):
    if output.exists():
        raise FileExistsError(f"Model directory exists; choose a new path: {output}")
    X, y, metadata = load_table(table)
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if len(np.unique(y)) < 2:
        raise ValueError("Context must include at least two different gameplay actions")
    print(f"Loading TabPFN v3.5; context={X.shape}, device={device}", flush=True)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import save_fitted_tabpfn_model

    kwargs = dict(
        device=device,
        n_estimators=n_estimators,
        fit_mode=fit_mode,
        random_state=seed,
        categorical_features_indices=list(CATEGORICAL_INDICES),
    )
    if model_path is not None:
        kwargs["model_path"] = str(model_path.resolve())
    model = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, **kwargs)
    started = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - started
    # Exercise predict before writing a successful model artifact. This timing
    # includes first-predict setup and is not a steady-state FPS estimate.
    started = time.perf_counter()
    prediction = int(model.predict(X[:1])[0])
    first_predict_seconds = time.perf_counter() - started
    validate_action(prediction)
    output.mkdir(parents=True)
    fitted_path = output / "policy.tabpfn_fit"
    save_fitted_tabpfn_model(model, fitted_path)
    manifest = {
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "model_version": "v3.5",
        "tabpfn_version": importlib.metadata.version("tabpfn"),
        "checkpoint_path": str(model.model_path),
        "context": metadata,
        "classes": [int(value) for value in model.classes_],
        "device_requested": device,
        "n_estimators": n_estimators,
        "fit_mode": fit_mode,
        "fit_seconds": fit_seconds,
        "first_predict_seconds": first_predict_seconds,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class Policy:
    def __init__(self, directory: Path, device="auto"):
        self.manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest["schema"] != SCHEMA or self.manifest["feature_names"] != list(
            FEATURE_NAMES
        ):
            raise ValueError("Model feature schema is incompatible with this code")
        installed = importlib.metadata.version("tabpfn")
        if installed != self.manifest["tabpfn_version"]:
            raise ValueError(
                f"Fitted policy requires tabpfn=={self.manifest['tabpfn_version']}; "
                f"installed {installed}. Refit or use the original package version."
            )
        fitted = directory / "policy.tabpfn_fit"
        from tabpfn.model_loading import load_fitted_tabpfn_model

        self.model = load_fitted_tabpfn_model(fitted, device=device)
        self._previous_ram = None

    def reset_history(self):
        """Forget temporal context at an episode boundary."""
        self._previous_ram = None

    def predict_ram(
        self, ram, *, previous_ram=None, action_value=1, selection="argmax", rng=None
    ):
        started = time.perf_counter()
        prior = self._previous_ram if previous_ram is None else previous_ram
        features = extract_features(ram, prior, action_value=action_value)
        self._previous_ram = checked_ram(ram).astype(np.uint8)
        probabilities = self.model.predict_proba(features[None, :])[0]
        if selection == "argmax":
            index = int(np.argmax(probabilities))
        elif selection == "sample":
            rng = np.random.default_rng() if rng is None else rng
            index = int(rng.choice(len(probabilities), p=probabilities))
        else:
            raise ValueError("selection must be argmax or sample")
        action = validate_action(int(self.model.classes_[index]))
        return {
            "action": action,
            "buttons": button_names(action),
            "confidence": float(probabilities[index]),
            "selection": selection,
            "desired_action_value": int(action_value),
            "predict_seconds": time.perf_counter() - started,
        }


def evaluate(policy: Policy, table: Path, batch_size=128):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
    )

    X, y, metadata = load_table(table)
    if metadata["label_offset"] != policy.manifest["context"]["label_offset"]:
        raise ValueError("Evaluation and context label offsets must agree")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    started = time.perf_counter()
    predictions = np.concatenate(
        [
            policy.model.predict(X[i : i + batch_size])
            for i in range(0, len(X), batch_size)
        ]
    )
    return {
        "rows": len(y),
        "seconds": time.perf_counter() - started,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "unseen_target_actions": sorted(
            set(map(int, y)) - set(policy.manifest["classes"])
        ),
        "classification_report": classification_report(
            y, predictions, output_dict=True, zero_division=0
        ),
    }
