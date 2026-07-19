"""Documented local dataset registration and canonical mapping."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from django.utils import timezone

from ml.data.dataset_registry import DATASETS
from ml.pipeline import CANONICAL_FEATURES
from .models import DatasetRegistry


def inspect_and_register_local_datasets() -> list[DatasetRegistry]:
    records = []
    for name, metadata in DATASETS.items():
        path = Path(metadata["path"])
        frame = pd.read_csv(path)
        mapping = metadata.get("feature_mappings", {})
        is_synthetic = metadata["kind"] == "derived_synthetic"
        record, _ = DatasetRegistry.objects.update_or_create(
            name=name,
            defaults={
                "description": metadata["source"],
                "source_url": metadata.get("source_url") or "",
                "license_name": metadata.get("license") or "Unknown",
                "license_url": metadata.get("license_url") or "",
                "dataset_type": "fitness",
                "cohort_type": metadata.get("population", ""),
                "participant_count": None,
                "record_count": len(frame),
                "feature_schema": {
                    column: str(dtype) for column, dtype in frame.dtypes.items()
                },
                "target_schema": {},
                "feature_mapping": mapping,
                "local_file_path": str(path.relative_to(path.parents[1])),
                "imported_at": timezone.now(),
                "is_real_data": metadata["kind"] == "externally_sourced_unverified_license",
                "is_synthetic": is_synthetic,
                "limitations": " ".join(metadata.get("limitations", [])),
                "citation": metadata.get("citation", ""),
            },
        )
        records.append(record)
    return records


def map_row(registry: DatasetRegistry, row: dict) -> tuple[dict, list[str]]:
    mapped, missing = {}, []
    for feature in CANONICAL_FEATURES:
        source_column = registry.feature_mapping.get(feature, feature)
        value = row.get(source_column)
        if source_column == "Height (m)" and value not in (None, ""):
            value = float(value) * 100
        if value is None or pd.isna(value):
            missing.append(feature)
        else:
            mapped[feature] = float(value)
    return mapped, missing
