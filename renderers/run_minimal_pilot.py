#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
INTERVENTION_PROTOCOL_HASH = hashlib.sha256((ROOT / "research/minimal_pilot/intervention_protocol.md").read_bytes()).hexdigest()
sys.path.insert(0, str(ROOT / "src"))
from eclt.interventions import protected_tokens  # noqa: E402
from eclt.caching import CacheIdentity, stable_hash as cache_hash, validate_cached_identity  # noqa: E402
from eclt.execution.identity_state import normalize_variant  # noqa: E402
from eclt.execution.manifests import append_record  # noqa: E402
from eclt.qualification import build_genuine_change_executions  # noqa: E402
from eclt.providers import (  # noqa: E402
    ProviderError,
    ProviderResponse,
    RetryableProviderError,
    call_anthropic,
    call_google,
    call_openai,
    parse_json_text,
)

REQUIRED = {
    "OPENAI_API_KEY": "generation and VANILLA decisions",
    "ANTHROPIC_API_KEY": "grounded decisions",
    "GOOGLE_API_KEY|GEMINI_API_KEY": "independent verification",
}
DIMENSIONS = ["POLITENESS_DIRECTNESS", "VERBOSITY_COMPRESSION", "SURFACE_PARAPHRASE", "AFFECT_WITHOUT_URGENCY"]
USAGE: list[dict[str, Any]] = []
PRICES_PER_MILLION = {"openai": (0.40, 1.60), "anthropic": (1.00, 5.00), "google": (1.50, 9.00)}
VARIANT_SCHEMA = {
    "type": "object",
    "properties": {"variant_text": {"type": "string"}, "edit_summary": {"type": "string"}},
    "required": ["variant_text", "edit_summary"],
    "additionalProperties": False,
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "accepted": {"type": "boolean"},
        "label_compatible": {"type": "boolean"},
        "bidirectional_entailment": {"type": "boolean"},
        "contradiction": {"type": "boolean"},
        "information_changed": {"type": "boolean"},
        "alternative_label_collision": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["accepted", "label_compatible", "bidirectional_entailment", "contradiction", "information_changed", "alternative_label_collision", "reason"],
    "additionalProperties": False,
}


def decision_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {
                "anyOf": [
                    {"type": "string", "enum": labels},
                    {"type": "null"},
                ]
            },
            "confidence": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "null"},
                ]
            },
            "review": {"type": "boolean"},
        },
        "required": ["label", "confidence", "review"],
        "additionalProperties": False,
    }


def missing_access() -> dict[str, str]:
    missing = {k: v for k, v in REQUIRED.items() if "|" not in k and not os.getenv(k)}
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        missing["GOOGLE_API_KEY|GEMINI_API_KEY"] = REQUIRED["GOOGLE_API_KEY|GEMINI_API_KEY"]
    return missing


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def call_retry(function: Callable[[], ProviderResponse], max_retries: int) -> tuple[ProviderResponse, int]:
    error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return function(), attempt
        except (RetryableProviderError, json.JSONDecodeError) as exc:
            error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 4))
    assert error is not None
    raise error


def call_json_retry(function: Callable[[], ProviderResponse], max_retries: int) -> tuple[dict[str, Any], ProviderResponse, int]:
    error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = function()
        except RetryableProviderError as exc:
            error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 4))
            continue
        try:
            return parse_json_text(response.text), response, attempt
        except (ProviderError, json.JSONDecodeError) as exc:
            error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 4))
    assert error is not None
    raise error


def record_response(handle, response: ProviderResponse, role: str, source_id: str, retry_count: int,
                    identity: CacheIdentity | None = None, provenance: dict[str, Any] | None = None) -> None:
    record = {
        "role": role, "source_id": source_id, "provider": response.provider, "model": response.model,
        "request_id": response.request_id, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms, "retry_count": retry_count, "raw": response.raw,
    }
    if identity is not None:
        record.update(vars(identity)); record["cache_key"] = identity.key
    if provenance:
        record.update(provenance)
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    USAGE.append({"provider": response.provider, "input_tokens": response.input_tokens or 0, "output_tokens": response.output_tokens or 0, "latency_ms": response.latency_ms})


def humanize(label: str) -> str:
    return label.replace("_", " ")


def labels_for_dataset(cohort: list[dict[str, str]], dataset: str) -> list[str]:
    labels = sorted({r["official_human_label"] for r in cohort if r["dataset"] == dataset})
    return labels


def connectivity(protocol: dict[str, Any], raw_handle) -> None:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    calls = [
        ("connectivity_openai", lambda: call_openai(protocol["generator"]["model"], 'Return {"ok":true}.', "connectivity", schema)),
        ("connectivity_google", lambda: call_google(protocol["verifier"]["model"], 'Return {"ok":true}.', schema)),
        ("connectivity_anthropic", lambda: call_anthropic(protocol["decision_pipelines"][1]["model"], 'Return {"ok":true}.', schema)),
    ]
    failures = []
    for role, function in calls:
        try:
            parsed, response, retries = call_json_retry(function, protocol["max_retries"])
            if parsed.get("ok") is not True:
                raise ProviderError(f"{role} returned unexpected payload")
            record_response(raw_handle, response, role, "connectivity", retries)
            print(f"{role}: ok ({response.latency_ms} ms)")
        except Exception as exc:
            failures.append((role, type(exc).__name__, str(exc)[:2000]))
            print(f"{role}: failed: {type(exc).__name__}: {str(exc)[:500]}", file=sys.stderr)
    if failures:
        raise SystemExit("connectivity completed with failures: " + json.dumps(failures))


def variant_prompt(row: dict[str, str], dimension: str) -> str:
    return f"""Create exactly one label-preserving rewrite for an intent-routing audit.
Dimension: {dimension}
Official label: {row['official_human_label']} ({humanize(row['official_human_label'])})
Source: {row['text']}
Preserve request type, action, object, polarity, entities, numbers, dates, currencies, identifiers, and all operational facts.
Do not add urgency, fraud, unauthorized, lost, stolen, blame, certainty, or another transaction type.
Return the requested JSON only."""


def generate_variant(protocol: dict[str, Any], row: dict[str, str], dimension: str) -> tuple[dict[str, Any], ProviderResponse, int]:
    prompt = variant_prompt(row, dimension)
    return call_json_retry(lambda: call_openai(protocol["generator"]["model"], prompt, "label_preserving_variant", VARIANT_SCHEMA), protocol["max_retries"])


def verifier_prompt(row: dict[str, str], variant: str, labels: list[str]) -> str:
    return f"""Independently verify whether a rewrite preserves a human-defined intent label.
Source: {row['text']}
Variant: {variant}
Expected label: {row['official_human_label']} ({humanize(row['official_human_label'])})
Alternative labels: {', '.join(labels)}
Accept only if both texts entail the same operational request and no intent-bearing information, polarity, object, action, entity, number, date, currency, identifier, fraud/loss/unauthorized status, or in-scope status changed.
Set accepted=true only when label_compatible and bidirectional_entailment are true and contradiction, information_changed, and alternative_label_collision are false."""


def verify_variant(protocol: dict[str, Any], row: dict[str, str], variant: str, labels: list[str]) -> tuple[dict[str, Any], ProviderResponse, int]:
    prompt = verifier_prompt(row, variant, labels)
    result, response, retries = call_json_retry(lambda: call_google(protocol["verifier"]["model"], prompt, VERIFY_SCHEMA), protocol["max_retries"])
    computed = bool(result.get("label_compatible") and result.get("bidirectional_entailment") and not result.get("contradiction") and not result.get("information_changed") and not result.get("alternative_label_collision"))
    result["accepted"] = bool(result.get("accepted") and computed)
    return result, response, retries


def validate_decision(value: dict[str, Any], labels: list[str]) -> None:
    confidence = value.get("confidence")
    confidence_ok = confidence is None or (
        isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1
    )
    if value.get("label") not in labels + [None] or not isinstance(value.get("review"), bool) or not confidence_ok:
        raise ProviderError("decision failed local schema validation")


def openai_decision_prompt(text: str, labels: list[str]) -> str:
    return f"Route the utterance to exactly one candidate label, or review if no label fits.\nLabels: {', '.join(labels)}\nUtterance: {text}\nReturn JSON only."


def decide_openai(protocol: dict[str, Any], text: str, labels: list[str]) -> tuple[dict[str, Any], ProviderResponse, int]:
    prompt = openai_decision_prompt(text, labels)
    value, response, retries = call_json_retry(lambda: call_openai(protocol["decision_pipelines"][0]["model"], prompt, "routing_decision", decision_schema(labels)), protocol["max_retries"])
    validate_decision(value, labels)
    return value, response, retries


def anthropic_decision_prompt(text: str, labels: list[str]) -> str:
    definitions = "\n".join(f"- {label}: {humanize(label)}" for label in labels)
    return f"""Classify the operational intent using the definitions below. Distinguish neighboring labels by the requested action, object and event status. If no label fits, return review.
Definitions:
{definitions}
Utterance: {text}
Return only JSON with exactly: label (one listed label or null), confidence (0 to 1 or null), review (boolean)."""


def decide_anthropic(protocol: dict[str, Any], text: str, labels: list[str]) -> tuple[dict[str, Any], ProviderResponse, int]:
    prompt = anthropic_decision_prompt(text, labels)
    value, response, retries = call_json_retry(
        lambda: call_anthropic(protocol["decision_pipelines"][1]["model"], prompt, decision_schema(labels)),
        protocol["max_retries"],
    )
    validate_decision(value, labels)
    return value, response, retries


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cohort() -> list[dict[str, str]]:
    return list(csv.DictReader((ROOT / "results/minimal_pilot/cohort/cohort.csv").open(encoding="utf-8")))


def genuine_change_plan(protocol: dict[str, Any], limit: int | None = None,
                        manifest: Path | None = None) -> list[dict[str, Any]]:
    specs = list(csv.DictReader(manifest.open(encoding="utf-8"))) if manifest else None
    return [item.as_dict() for item in build_genuine_change_executions(
        protocol, _cohort(), limit=limit, control_specs=specs)]


def _control_identity(protocol: dict[str, Any], item: dict[str, Any], labels: list[str]) -> CacheIdentity:
    return CacheIdentity(
        item["provider"], item["model"], item["model_version"], "decision_changing_control",
        item["pipeline_condition"], item["prompt_hash"], item["schema_hash"], item["dataset"],
        item["split"], item["source_id"], item["source_text_hash"],
        protocol["genuine_change"]["control_contract_hash"], "HUMAN_LABELED_DONOR",
        item["control_text_hash"], cache_hash(labels), f"pilot-{protocol['version']}",
    )


def _decision_state(value: dict[str, Any]) -> str:
    if value.get("review") is True:
        return "HUMAN_REVIEW"
    label = value.get("label")
    if label is None:
        return "MISSING_DECISION"
    return "OOS" if str(label).upper() == "OOS" else "IN_SCOPE"


def run_genuine_change(protocol: dict[str, Any], limit: int, output_dir: Path, *, resume: bool = False,
                       manifest: Path | None = None) -> None:
    cohort = _cohort()
    plan = genuine_change_plan(protocol, limit, manifest)
    row_by_id = {row["source_row_id"]: row for row in cohort}
    labels_by_dataset = {dataset: labels_for_dataset(cohort, dataset) for dataset in {row["dataset"] for row in cohort}}
    result_path = output_dir / "genuine_change_results.jsonl"
    raw_path = output_dir / "raw_responses.jsonl"
    previous = read_jsonl(result_path) if resume else []
    completed = {row["execution_identity"]: row for row in previous}
    raw_cache = {row["cache_key"]: row for row in read_jsonl(raw_path) if isinstance(row.get("cache_key"), str)} if resume else {}
    failures: list[dict[str, str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as raw_handle:
        for index, item in enumerate(plan, 1):
            key = item["execution_identity"]
            if key in completed:
                continue
            source = row_by_id[item["source_id"]]
            donor = row_by_id[item["human_labeled_reference"]]
            labels = labels_by_dataset[item["dataset"]]
            identity = _control_identity(protocol, item, labels)
            assert identity.key == key
            try:
                cached = raw_cache.get(key)
                if cached is not None:
                    valid, reason = validate_cached_identity(identity, cached)
                    if not valid:
                        raise ProviderError(f"stale cache rejected: {reason}")
                    value = parse_json_text(raw_response_text(cached))
                    latency = cached.get("latency_ms")
                    retries = cached.get("retry_count", 0)
                    input_tokens = cached.get("input_tokens")
                    output_tokens = cached.get("output_tokens")
                    estimated_cost = cached.get("estimated_cost_usd")
                else:
                    function = decide_openai if item["provider"] == "openai" else decide_anthropic
                    value, response, retries = function(protocol, donor["text"], labels)
                    input_tokens, output_tokens, latency = response.input_tokens, response.output_tokens, response.latency_ms
                    input_price, output_price = PRICES_PER_MILLION[item["provider"]]
                    estimated_cost = ((input_tokens or 0) * input_price + (output_tokens or 0) * output_price) / 1_000_000
                    provenance = {
                        "official_split": item["split"], "source_text_hash": item["source_text_hash"],
                        "official_label": item["source_label"], "target_label": item["target_label"],
                        "control_text_hash": item["control_text_hash"], "control_construction_type": item["control_construction_type"],
                        "human_labeled_reference": item["human_labeled_reference"],
                        "changed_intent_bearing_concept": item["changed_intent_bearing_concept"],
                        "execution_date": datetime.now(timezone.utc).date().isoformat(),
                        "estimated_cost_usd": estimated_cost, "parse_status": "valid", "validation_status": "valid",
                    }
                    record_response(raw_handle, response, "decision_changing_control", item["source_id"], retries, identity, provenance)
                validate_decision(value, labels)
                state = _decision_state(value)
                normalized_prediction = "HUMAN_REVIEW" if state == "HUMAN_REVIEW" else (
                    "MISSING_DECISION" if state == "MISSING_DECISION" else value.get("label"))
                item.update({
                    "model_decision": value.get("label"),
                    "review_oos_state": state,
                    "parse_status": "VALID",
                    "latency_ms": latency,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost_usd": estimated_cost,
                    "retry_count": retries,
                    "target_rerouted": value.get("label") == item["target_label"] and value.get("review") is False,
                    "test_type": "changing_control",
                    "eligible": True,
                    "official_label": item["source_label"],
                    "control_prediction": normalized_prediction,
                    "condition": item["pipeline_condition"],
                    "pipeline": f"{item['provider']}:{item['model']}",
                    "intervention_dimension": "HUMAN_LABELED_DONOR",
                })
                completed[key] = item
                print(f"genuine-change {index}/{len(plan)}: {item['provider']} {item['source_id']}")
            except Exception as exc:
                failures.append({"execution_identity": key, "error_type": type(exc).__name__, "error": str(exc)[:2000]})
    rows = [completed[item["execution_identity"]] for item in plan if item["execution_identity"] in completed]
    result_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output_dir / "genuine_change_failures.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")
    summary = {
        "protocol_version": protocol["version"], "planned_decisions": len(plan), "completed_decisions": len(rows),
        "failures": len(failures), "source_controls": len({row["source_id"] for row in rows}),
        "target_rerouting_rate": (sum(bool(row["target_rerouted"]) for row in rows) / len(rows)) if rows else None,
        "review_or_oos_rate": (sum(row["review_oos_state"] != "IN_SCOPE" for row in rows) / len(rows)) if rows else None,
        "estimated_total_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0) for row in rows), 6),
    }
    (output_dir / "genuine_change_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("genuine-change execution completed with failures")


def raw_response_text(record: dict[str, Any]) -> str:
    body = record["raw"]
    if record["provider"] == "openai":
        if isinstance(body.get("output_text"), str):
            return body["output_text"]
        return "".join(
            content.get("text", "")
            for item in body.get("output", [])
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )
    if record["provider"] == "google":
        return "".join(part.get("text", "") for part in body["candidates"][0]["content"]["parts"])
    if record["provider"] == "anthropic":
        return "".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")
    raise ProviderError(f"unsupported cached provider: {record['provider']}")


def run_smoke(protocol: dict[str, Any], limit: int, raw_handle, output_dir: Path, resume: bool = False,
              manifest: Path | None = None) -> None:
    cohort = list(csv.DictReader((ROOT / "results/minimal_pilot/cohort/cohort.csv").open(encoding="utf-8")))
    manifest_rows = list(csv.DictReader(manifest.open(encoding="utf-8"))) if manifest else []
    if manifest_rows:
        by_id = {row["source_row_id"]: row for row in cohort}
        target = [by_id[row["source_id"]] for row in manifest_rows[:limit]]
        dimension_by_id = {row["source_id"]: row["same_intent_dimension"] for row in manifest_rows}
    else:
        target = cohort[:limit]
        dimension_by_id = {}
    result_path = output_dir / "smoke_results.jsonl"
    summary_path = output_dir / "smoke_summary.json"
    previous_rows = read_jsonl(result_path) if resume else []
    variant_manifest_path = output_dir / "variant_manifest.jsonl"
    prior_variants = {
        row["source_id"]: row for row in read_jsonl(variant_manifest_path)
        if isinstance(row.get("source_id"), str)
    }
    raw_cache: dict[str, dict[str, Any]] = {}
    if resume:
        for record in read_jsonl(output_dir / "raw_responses.jsonl"):
            if isinstance(record.get("cache_key"), str):
                raw_cache[str(record["cache_key"])] = record
    completed = {row["source_id"]: row for row in previous_rows}
    selected = [(index, row) for index, row in enumerate(target) if row["source_row_id"] not in completed]
    if resume:
        print(f"resuming smoke: {len(completed)}/{limit} completed; {len(selected)} remaining")
    failures: list[dict[str, str]] = []
    execution_dimensions = protocol.get("stage1_design", {}).get("same_intent_dimensions", DIMENSIONS)
    for index, row in selected:
        dimension = dimension_by_id.get(row["source_row_id"], execution_dimensions[index % len(execution_dimensions)])
        labels = labels_for_dataset(cohort, row["dataset"])
        try:
            source_id = row["source_row_id"]

            base_provenance = {
                "official_split": row["official_split"], "source_text_hash": row["text_hash"],
                "official_label": row["official_human_label"], "label_risk_stratum": row["label_noise_risk"],
                "duplicate_component_id": row["duplicate_component_id"], "execution_date": datetime.now(timezone.utc).date().isoformat(),
                "estimated_cost_usd": None, "parse_status": "valid", "validation_status": "valid",
            }

            def identity_for(role: str, provider: str, model: str, condition: str, prompt: str,
                             schema: dict[str, Any], variant_text: str,
                             parent_identity_hash: str = "") -> CacheIdentity:
                return CacheIdentity(provider, model, model, role, condition, cache_hash(prompt), cache_hash(schema), row["dataset"],
                                     row["official_split"], source_id, row["text_hash"], INTERVENTION_PROTOCOL_HASH,
                                     dimension, "" if role == "GENERATOR" else cache_hash(variant_text),
                                     cache_hash(labels), f"pilot-{protocol['version']}", parent_identity_hash)

            def cached_or_call(role: str, identity: CacheIdentity, function: Callable[[], tuple[dict[str, Any], ProviderResponse, int]]) -> dict[str, Any]:
                cached = raw_cache.get(identity.key)
                if cached is not None:
                    valid, reason = validate_cached_identity(identity, cached)
                    if not valid:
                        raise ProviderError(f"stale cache rejected: {reason}")
                    print(f"smoke {index + 1}/{limit}: reusing {role}")
                    return parse_json_text(raw_response_text(cached))
                value, response, retries = function()
                record_response(raw_handle, response, role, source_id, retries, identity, base_provenance)
                return value

            generator_identity = identity_for("GENERATOR", "openai", protocol["generator"]["model"], "GENERATION",
                                              variant_prompt(row, dimension), VARIANT_SCHEMA, "")
            variant_result = cached_or_call("generator", generator_identity, lambda: generate_variant(protocol, row, dimension))
            raw_variant = str(variant_result["variant_text"])
            variant = normalize_variant(raw_variant)
            variant_record = {
                "source_id": source_id, "raw_generated_text": raw_variant, "normalized_text": variant,
                "variant_hash": cache_hash(variant), "generation_identity_hash": generator_identity.key,
                "variant_manifest_hash": cache_hash({
                    "source_id": source_id, "variant_hash": cache_hash(variant),
                    "generation_identity_hash": generator_identity.key,
                }),
                "status": "VARIANT_MATERIALIZED",
            }
            prior_variant = prior_variants.get(source_id)
            if prior_variant is not None and prior_variant != variant_record:
                raise ProviderError("variant manifest mismatch for resumed source")
            if prior_variant is None:
                append_record(variant_manifest_path, variant_record)
                prior_variants[source_id] = variant_record
            local_tokens_ok = protected_tokens(row["text"]) == protected_tokens(variant)
            verifier_identity = identity_for(
                "VERIFIER", "google", protocol["verifier"]["model"], "INDEPENDENT_VERIFICATION",
                verifier_prompt(row, variant, labels),
                {
                    "response_schema": VERIFY_SCHEMA,
                    "generation_config": {"thinking_level": "low"},
                },
                variant, generator_identity.key)
            verification = cached_or_call("verifier", verifier_identity, lambda: verify_variant(protocol, row, variant, labels))
            computed_verification = bool(verification.get("label_compatible") and verification.get("bidirectional_entailment") and not verification.get("contradiction") and not verification.get("information_changed") and not verification.get("alternative_label_collision"))
            verification["accepted"] = bool(verification.get("accepted") and computed_verification)
            accepted = bool(local_tokens_ok and verification["accepted"] and variant and variant != row["text"])
            result: dict[str, Any] = {
                "source_id": row["source_row_id"], "dataset": row["dataset"], "official_label": row["official_human_label"],
                "dimension": dimension, "source_text": row["text"], "variant_text": variant, "accepted": accepted,
                "local_protected_tokens_ok": local_tokens_ok, "verification_reason": verification.get("reason"),
            }
            for name, function in (("openai", decide_openai), ("anthropic", decide_anthropic)):
                spec = protocol["decision_pipelines"][0 if name == "openai" else 1]
                prompt_function = openai_decision_prompt if name == "openai" else anthropic_decision_prompt
                source_identity = identity_for(f"decision_{name}_source", spec["provider"], spec["model"], spec["condition"],
                                               prompt_function(row["text"], labels), decision_schema(labels), row["text"])
                source_decision = cached_or_call(f"decision_{name}_source", source_identity, lambda: function(protocol, row["text"], labels))
                validate_decision(source_decision, labels)
                result[f"{name}_source_label"] = source_decision.get("label")
                result[f"{name}_source_confidence"] = source_decision.get("confidence")
                if accepted:
                    variant_identity = identity_for(f"decision_{name}_variant", spec["provider"], spec["model"], spec["condition"],
                                                    prompt_function(variant, labels), decision_schema(labels), variant)
                    variant_decision = cached_or_call(f"decision_{name}_variant", variant_identity, lambda: function(protocol, variant, labels))
                    validate_decision(variant_decision, labels)
                    result[f"{name}_variant_label"] = variant_decision.get("label")
                    result[f"{name}_variant_confidence"] = variant_decision.get("confidence")
            completed[row["source_row_id"]] = result
            print(f"smoke {index + 1}/{limit}: {row['dataset']} {dimension} accepted={accepted}")
        except Exception as exc:
            failures.append({"source_id": row["source_row_id"], "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            print(f"smoke {index + 1}/{limit}: failed: {type(exc).__name__}", file=sys.stderr)
    rows = [completed[row["source_row_id"]] for row in target if row["source_row_id"] in completed]
    result_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    (output_dir / "smoke_failures.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in failures), encoding="utf-8")
    prior_usage = {}
    if resume and summary_path.exists():
        prior_usage = json.loads(summary_path.read_text(encoding="utf-8")).get("provider_usage", {})
    provider_usage = {}
    estimated_cost = 0.0
    for provider in sorted(PRICES_PER_MILLION):
        input_tokens = sum(x["input_tokens"] for x in USAGE if x["provider"] == provider) + prior_usage.get(provider, {}).get("input_tokens", 0)
        output_tokens = sum(x["output_tokens"] for x in USAGE if x["provider"] == provider) + prior_usage.get(provider, {}).get("output_tokens", 0)
        input_price, output_price = PRICES_PER_MILLION[provider]
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        estimated_cost += cost
        provider_usage[provider] = {"input_tokens": input_tokens, "output_tokens": output_tokens, "estimated_cost_usd": round(cost, 6)}
    summary = {
        "protocol_version": protocol["version"], "attempted_sources": limit, "completed_sources": len(rows), "failures": len(failures),
        "accepted_variants": sum(bool(x["accepted"]) for x in rows), "inference_eligible": False,
        "provider_usage": provider_usage, "estimated_total_cost_usd": round(estimated_cost, 6),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit("smoke completed with failures; inspect smoke_failures.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--connectivity", action="store_true")
    parser.add_argument("--smoke", type=int, metavar="N")
    parser.add_argument("--resume-smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--genuine-change", type=int, metavar="N")
    parser.add_argument("--genuine-change-manifest", type=Path)
    parser.add_argument("--resume-genuine-change", action="store_true")
    parser.add_argument("--dry-run-genuine-change", type=int, metavar="N")
    parser.add_argument("--dry-run-output", type=Path)
    parser.add_argument("--stage1-cohort-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("version") not in {"2.2", "2.2-amendment-1"}:
        raise SystemExit("BLOCKED: unsupported protocol version")
    if args.dry_run_genuine_change is not None:
        if protocol.get("version") != "2.2-amendment-1":
            raise SystemExit("BLOCKED: genuine-change dry run requires amended protocol")
        if not 1 <= args.dry_run_genuine_change <= 450:
            raise SystemExit("dry-run genuine-change N must be in [1, 450]")
        plan = genuine_change_plan(protocol, args.dry_run_genuine_change, args.genuine_change_manifest)
        output = args.dry_run_output or ROOT / "results/protocol_amendment/genuine_change_execution_plan.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plan), encoding="utf-8")
        print(json.dumps({"planned_decisions": len(plan), "source_controls": args.dry_run_genuine_change,
                          "external_calls": 0, "output": str(output)}, indent=2))
        return
    missing = missing_access()
    if missing:
        raise SystemExit("BLOCKED: missing model access: " + json.dumps(missing, sort_keys=True))
    if not args.execute:
        print("access configuration detected; pass --execute for external calls")
        return
    if args.full:
        raise SystemExit("BLOCKED: full 450-source execution requires a separate post-smoke approval and implementation review")
    if not args.connectivity and args.smoke is None and args.genuine_change is None:
        raise SystemExit("choose --connectivity, --smoke N, or --genuine-change N")
    if args.smoke is not None and not 1 <= args.smoke <= 30:
        raise SystemExit("smoke N must be in [1, 30]")
    if args.resume_smoke and args.smoke is None:
        raise SystemExit("--resume-smoke requires --smoke N")
    if args.genuine_change is not None:
        if protocol.get("version") != "2.2-amendment-1":
            raise SystemExit("BLOCKED: genuine-change execution requires amended protocol")
        if not 1 <= args.genuine_change <= 450:
            raise SystemExit("genuine-change N must be in [1, 450]")
    if args.resume_genuine_change and args.genuine_change is None:
        raise SystemExit("--resume-genuine-change requires --genuine-change N")
    if args.stage1_cohort_manifest is not None and args.smoke is None:
        raise SystemExit("--stage1-cohort-manifest requires --smoke N")
    if args.genuine_change_manifest is not None and args.genuine_change is None and args.dry_run_genuine_change is None:
        raise SystemExit("--genuine-change-manifest requires a genuine-change stage")
    output_dir = args.output_dir or ROOT / "results/minimal_pilot/smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.connectivity or args.smoke is not None:
        with (output_dir / "raw_responses.jsonl").open("a", encoding="utf-8") as raw_handle:
            if args.connectivity:
                connectivity(protocol, raw_handle)
            if args.smoke is not None:
                run_smoke(
                    protocol,
                    args.smoke,
                    raw_handle,
                    output_dir,
                    resume=args.resume_smoke,
                    manifest=args.stage1_cohort_manifest,
                )
    if args.genuine_change is not None:
        run_genuine_change(protocol, args.genuine_change,
                           args.output_dir or ROOT / "results/minimal_pilot/genuine_change",
                           resume=args.resume_genuine_change, manifest=args.genuine_change_manifest)


if __name__ == "__main__":
    main()
