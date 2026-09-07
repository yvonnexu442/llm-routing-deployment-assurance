from __future__ import annotations

from collections import defaultdict

from eclt.caching import CacheIdentity, stable_hash
from eclt.interventions import PositiveControl

from .execution import PlannedDecision


def _decision_schema(labels: list[str]) -> dict[str, object]:
    return {"type": "object", "properties": {
        "label": {"anyOf": [{"type": "string", "enum": labels}, {"type": "null"}]},
        "confidence": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "review": {"type": "boolean"}}, "required": ["label", "confidence", "review"], "additionalProperties": False}


def _decision_prompt(condition: str, text: str, labels: list[str]) -> str:
    if condition == "VANILLA":
        return f"Route the utterance to exactly one candidate label, or review if no label fits.\nLabels: {', '.join(labels)}\nUtterance: {text}\nReturn JSON only."
    definitions = "\n".join(f"- {label}: {label.replace('_', ' ')}" for label in labels)
    return f"""Classify the operational intent using the definitions below. Distinguish neighboring labels by the requested action, object and event status. If no label fits, return review.
Definitions:
{definitions}
Utterance: {text}
Return only JSON with exactly: label (one listed label or null), confidence (0 to 1 or null), review (boolean)."""


def build_human_donor_controls(rows: list[dict[str, str]]) -> list[PositiveControl]:
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows: by_dataset[row["dataset"]].append(row)
    controls = []
    for row in rows:
        donors = sorted((candidate for candidate in by_dataset[row["dataset"]]
                         if candidate["official_human_label"] != row["official_human_label"] and candidate["source_row_id"] != row["source_row_id"]),
                        key=lambda candidate: (candidate["official_human_label"], candidate["source_row_id"]))
        if not donors: continue
        donor = donors[0]
        control = PositiveControl(row["source_row_id"], donor["source_row_id"], row["official_human_label"],
                                  donor["official_human_label"], "human-labeled donor intent", donor["text_hash"])
        if control.validate()[0]: controls.append(control)
    return controls


def planned_identity(protocol: dict[str, object], row: dict[str, str], *, provider: str, model: str, role: str,
                     condition: str, prompt_spec: object, schema_spec: object, dimension: str, variant_hash: str,
                     label_set: list[str]) -> CacheIdentity:
    intervention_hash = str(protocol.get("_intervention_protocol_hash") or stable_hash(protocol["primary_dimensions"]))
    return CacheIdentity(provider, model, model, role, condition, stable_hash(prompt_spec), stable_hash(schema_spec), row["dataset"],
                         row["official_split"], row["source_row_id"], row["text_hash"], intervention_hash,
                         dimension, variant_hash, stable_hash(label_set),
                         f"pilot-{protocol['version']}")


def build_decision_plan(protocol: dict[str, object], rows: list[dict[str, str]]) -> list[PlannedDecision]:
    plan = []
    labels_by_dataset = {dataset: sorted({row["official_human_label"] for row in rows if row["dataset"] == dataset})
                         for dataset in {row["dataset"] for row in rows}}
    for row in rows:
        for spec in protocol["decision_pipelines"]:  # type: ignore[index]
            labels = labels_by_dataset[row["dataset"]]
            role = f"decision_{spec['provider']}_source"
            identity = planned_identity(protocol, row, provider=spec["provider"], model=spec["model"], role=role,
                                        condition=spec["condition"], prompt_spec=_decision_prompt(spec["condition"], row["text"], labels),
                                        schema_spec=_decision_schema(labels), dimension="SOURCE", variant_hash=row["text_hash"], label_set=labels)
            plan.append(PlannedDecision(identity, "source", row["official_human_label"]))
    row_by_id = {row["source_row_id"]: row for row in rows}
    for row, control in zip(rows, build_human_donor_controls(rows)):
        donor = row_by_id[control.donor_id]
        for spec in protocol["decision_pipelines"]:  # type: ignore[index]
            labels = labels_by_dataset[row["dataset"]]
            identity = planned_identity(protocol, row, provider=spec["provider"], model=spec["model"], role="decision_changing_control",
                                        condition=spec["condition"], prompt_spec=_decision_prompt(spec["condition"], donor["text"], labels),
                                        schema_spec=_decision_schema(labels), dimension="HUMAN_LABELED_DONOR",
                                        variant_hash=control.donor_text_hash, label_set=labels)
            plan.append(PlannedDecision(identity, "changing_control", row["official_human_label"], control.target_label))
    for item in plan: item.validate()
    return plan
