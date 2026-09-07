#!/usr/bin/env python3
"""Stage-2B-only sequential executor for the frozen 160-source expansion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eclt.budget import BudgetBlocked, BudgetGuard, PriceTable  # noqa: E402
from eclt.budget.ledger import CostLedger  # noqa: E402
from eclt.caching import CacheIdentity, stable_hash  # noqa: E402
from eclt.execution.identity_state import normalize_variant  # noqa: E402
from eclt.execution.manifests import append_record, write_manifest_atomic  # noqa: E402
from eclt.interventions import protected_tokens  # noqa: E402
from eclt.providers import (  # noqa: E402
    ProviderError, ProviderResponse, RetryableProviderError,
    call_anthropic, call_google, call_openai, parse_json_text,
)
from run_minimal_pilot import (  # noqa: E402
    VARIANT_SCHEMA, VERIFY_SCHEMA, anthropic_decision_prompt, decision_schema,
    openai_decision_prompt, variant_prompt, verifier_prompt,
)

BRANCH = "simplified-deployment-readiness-thesis"
STAGE_TOKEN = "STAGE2B"
STAGE_NAME = "2B"
DATASET = "BANKING77"
EXECUTION_VERSION = "stage2b-1.0"
EXPECTED_SOURCE_COUNT = 160
EXPECTED_LABEL_COUNT = 77
TARGET_TOTAL_SOURCES = 200
NEXT_STAGE = "2C"
DESIGN = ROOT / "research" / "formal_experiments" / "stage2b"
RUNTIME = ROOT / "results" / "stage2b" / "runtime"
PREFLIGHT = RUNTIME / "preflight.json"
COMPLETIONS = RUNTIME / "completed.jsonl"
FAILURES = RUNTIME / "failures.jsonl"
STATE = RUNTIME / "state.json"
LEDGER = RUNTIME / "cost_ledger.csv"
AUTH_ENV = "ECLT_STAGE2B_AUTHORIZED"
AUTH_VALUE = "YES"
ADAPTER_BUDGET_ENV = (
    "ECLT_BUDGET_PRICING", "ECLT_BUDGET_LEDGER", "ECLT_BUDGET_CEILING",
    "ECLT_BUDGET_WARNING", "ECLT_BUDGET_RETRY_RESERVE",
    "ECLT_REQUIRE_BUDGET_GUARD",
)
BASE_CALLS = 2240
RETRY_RESERVE = 224
MAX_ATTEMPTS = 2464
WARNING_USD = 3.0
CEILING_USD = 3.5
MAX_JSON_RETRIES = 2
VERIFIER_BOOLEAN_FIELDS = (
    "accepted", "label_compatible", "bidirectional_entailment", "contradiction",
    "information_changed", "alternative_label_collision",
)
INTERVENTION_HASH = hashlib.sha256(
    (ROOT / "research" / "minimal_pilot" / "intervention_protocol.md").read_bytes()
).hexdigest()
CELLS = {
    "O_V": ("openai", "gpt-4.1-mini-2025-04-14", "VANILLA"),
    "O_G": ("openai", "gpt-4.1-mini-2025-04-14", "LABEL_DEFINITION_GROUNDED"),
    "A_V": ("anthropic", "claude-haiku-4-5-20251001", "VANILLA"),
    "A_G": ("anthropic", "claude-haiku-4-5-20251001", "LABEL_DEFINITION_GROUNDED"),
}
HASHED_FILES = (
    "cohort_manifest.json", "cohort_plan.csv", "genuine_change_control_plan.csv",
    "call_plan.csv", "projected_cost.json", "execution_stop_rules.md",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design_hashes() -> dict[str, str]:
    return {name: sha(DESIGN / name) for name in HASHED_FILES}


@lru_cache(maxsize=1)
def inputs() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]], list[str]]:
    cohort = read_csv(DESIGN / "cohort_plan.csv")
    controls = {row["source_id"]: row for row in read_csv(DESIGN / "genuine_change_control_plan.csv")}
    dataset = {
        row["source_row_id"]: row
        for row in read_jsonl(ROOT / "data" / "processed" / "minimal_pilot_rows.jsonl")
        if row["dataset"] == DATASET and row["official_split"] == "test"
    }
    labels = sorted({row["official_human_label"] for row in dataset.values()})
    if (
        len(cohort) != EXPECTED_SOURCE_COUNT
        or len(controls) != EXPECTED_SOURCE_COUNT
        or len(labels) != EXPECTED_LABEL_COUNT
    ):
        raise ValueError(f"frozen Stage {STAGE_NAME} input cardinality mismatch")
    return cohort, controls, dataset, labels


def current_state() -> dict[str, int]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"attempt_count": 0, "failed_attempts": 0}


def completed() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(COMPLETIONS)
    keys = [row["cache_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate completed cache identity")
    return {row["cache_key"]: row for row in rows}


def identity(
    row: dict[str, str], *, provider: str, model: str, role: str, condition: str,
    prompt: str, schema: dict[str, Any], dimension: str, variant_hash: str,
    contract_hash: str = INTERVENTION_HASH, parent_identity_hash: str = "",
) -> CacheIdentity:
    _, _, _, labels = inputs()
    return CacheIdentity(
        provider, model, model, role, condition, stable_hash(prompt), stable_hash(schema),
        DATASET, row.get("split", "test"), row["source_id"], row["source_text_hash"], contract_hash,
        dimension, variant_hash, stable_hash(labels), EXECUTION_VERSION, parent_identity_hash,
    )


def validate_design() -> dict[str, bool]:
    cohort, controls, dataset, labels = inputs()
    source_ids = {row["source_id"] for row in cohort}
    checks = {
        "source_count": len(source_ids) == EXPECTED_SOURCE_COUNT,
        "source_hashes": all(dataset[row["source_id"]]["text_hash"] == row["source_text_hash"] for row in cohort),
        "control_hashes": all(
            dataset[controls[source]["donor_id"]]["text_hash"] == controls[source]["control_text_hash"]
            for source in source_ids
        ),
        "four_cells": set(CELLS) == {"O_V", "O_G", "A_V", "A_G"},
        "labels": len(labels) == EXPECTED_LABEL_COUNT,
        "base_calls": BASE_CALLS == EXPECTED_SOURCE_COUNT * 14,
        f"no_stage{NEXT_STAGE.lower()}": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage {STAGE_NAME} frozen design validation failed")
    return checks


def preflight(pricing: Path) -> str:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    try:
        validate_design()
        checks["manifest"] = True
    except Exception:
        checks["manifest"] = False
    checks["repository"] = Path.cwd().resolve() == ROOT
    checks["branch"] = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip() == BRANCH
    checks["clean_worktree"] = not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    try:
        table = PriceTable.from_file(pricing)
        required = {
            ("openai", "gpt-4.1-mini-2025-04-14"),
            ("anthropic", "claude-haiku-4-5-20251001"),
            ("google", "gemini-3.5-flash"),
        }
        checks["pricing"] = required <= table.identities()
    except Exception:
        checks["pricing"] = False
    checks["providers"] = all(bool(os.environ.get(name)) for name in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    )) and bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    projection = json.loads((DESIGN / "projected_cost.json").read_text(encoding="utf-8"))
    checks["budget"] = (
        projection["maximum_base_calls"] == BASE_CALLS
        and projection["maximum_attempts"] == MAX_ATTEMPTS
        and projection["hard_cost_ceiling_usd"] == CEILING_USD
    )
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    checks["tests"] = test.returncode == 0
    if all(checks.values()):
        status = f"{STAGE_TOKEN}_PREFLIGHT_PASS"
    elif not checks["manifest"]:
        status = f"{STAGE_TOKEN}_PREFLIGHT_BLOCKED_MANIFEST"
    elif not checks["providers"]:
        status = f"{STAGE_TOKEN}_PREFLIGHT_BLOCKED_PROVIDER"
    elif not checks["pricing"]:
        status = f"{STAGE_TOKEN}_PREFLIGHT_BLOCKED_PRICING"
    elif not checks["budget"]:
        status = f"{STAGE_TOKEN}_PREFLIGHT_BLOCKED_BUDGET"
    else:
        status = f"{STAGE_TOKEN}_PREFLIGHT_BLOCKED_IMPLEMENTATION"
    write_manifest_atomic(PREFLIGHT, {
        "status": status, "provider_calls": 0, "checks": checks,
        "design_hashes": design_hashes(), "pricing_hash": sha(pricing),
        "pricing_path": str(pricing.resolve()), "base_calls": BASE_CALLS,
        "retry_reserve": RETRY_RESERVE, "maximum_attempts": MAX_ATTEMPTS,
        "expected_cost_usd": 2.59, "warning_usd": WARNING_USD,
        "hard_ceiling_usd": CEILING_USD,
        "test_summary": test.stdout.strip().splitlines()[-1] if test.stdout.strip() else "",
    })
    print(status)
    return status


def require_preflight(pricing: Path) -> None:
    if not PREFLIGHT.exists():
        raise RuntimeError("Stage 2B preflight is missing")
    value = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if value["status"] != f"{STAGE_TOKEN}_PREFLIGHT_PASS":
        raise RuntimeError(f"latest Stage {STAGE_NAME} preflight is not PASS")
    if value["design_hashes"] != design_hashes() or value["pricing_hash"] != sha(pricing):
        raise RuntimeError(f"Stage {STAGE_NAME} design or pricing changed after preflight")


def recover_verifier_json(text: str) -> dict[str, Any]:
    """Recover only exact, unique verifier booleans from otherwise malformed JSON."""
    recovered: dict[str, Any] = {}
    for field in VERIFIER_BOOLEAN_FIELDS:
        matches = re.findall(rf'"{re.escape(field)}"\s*:\s*(true|false)', text)
        if len(matches) != 1:
            raise json.JSONDecodeError(
                f"cannot strictly recover verifier field {field}", text, 0
            )
        recovered[field] = matches[0] == "true"
    recovered["reason"] = "STRUCTURED_OUTPUT_RECOVERED_BOOLEAN_FIELDS"
    return recovered


def perform(
    cache: dict[str, dict[str, Any]], guard: BudgetGuard, state: dict[str, int],
    planned: CacheIdentity, call: Callable[[], ProviderResponse],
) -> dict[str, Any]:
    if planned.key in cache:
        observed = cache[planned.key]
        if any(observed.get(name) != value for name, value in vars(planned).items()):
            raise RuntimeError("stale completed identity")
        return observed["value"]
    if state["attempt_count"] >= MAX_ATTEMPTS or state["failed_attempts"] >= RETRY_RESERVE:
        raise RuntimeError(f"Stage {STAGE_NAME} attempt reserve exhausted")
    authorization = guard.authorize(
        planned.provider, planned.model, planned.role,
        json.dumps(vars(planned), sort_keys=True),
    )
    state["attempt_count"] += 1
    write_manifest_atomic(STATE, state)
    response = None
    parse_recovery = ""
    try:
        response = call()
        try:
            value = parse_json_text(response.text)
        except json.JSONDecodeError:
            if planned.role != "VERIFIER":
                raise
            value = recover_verifier_json(response.text)
            parse_recovery = "STRICT_VERIFIER_BOOLEAN_RECOVERY"
        guard.record_actual(
            authorization,
            input_tokens=response.input_tokens or authorization.estimate.estimated_input_tokens,
            output_tokens=response.output_tokens or authorization.estimate.reserved_output_tokens,
            request_id=response.request_id,
        )
        record = {
            **vars(planned), "cache_key": planned.key, "value": value,
            "request_id": response.request_id, "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens, "latency_ms": response.latency_ms,
            "parse_recovery": parse_recovery,
        }
        append_record(COMPLETIONS, record)
        cache[planned.key] = record
        return value
    except (ProviderError, RetryableProviderError, json.JSONDecodeError, BudgetBlocked) as exc:
        budget_error = None
        try:
            guard.record_actual(
                authorization,
                input_tokens=(
                    response.input_tokens
                    if response and response.input_tokens
                    else authorization.estimate.estimated_input_tokens
                ),
                output_tokens=(
                    response.output_tokens
                    if response and response.output_tokens
                    else authorization.estimate.reserved_output_tokens
                ),
                request_id=response.request_id if response else None,
            )
        except BudgetBlocked as blocked:
            budget_error = blocked
        state["failed_attempts"] += 1
        write_manifest_atomic(STATE, state)
        append_record(FAILURES, {
            "cache_key": planned.key, "source_id": planned.source_id,
            "role": planned.role, "error_type": type(exc).__name__, "error": str(exc)[:1000],
        })
        if budget_error is not None:
            raise budget_error from exc
        raise


def perform_json_retry(
    cache: dict[str, dict[str, Any]], guard: BudgetGuard, state: dict[str, int],
    planned: CacheIdentity, call: Callable[[], ProviderResponse],
) -> dict[str, Any]:
    """Retry transient or malformed output; each attempt remains budgeted and logged."""
    for retry in range(MAX_JSON_RETRIES + 1):
        try:
            return perform(cache, guard, state, planned, call)
        except (json.JSONDecodeError, RetryableProviderError) as exc:
            if retry == MAX_JSON_RETRIES:
                raise
            print(
                f"stage{STAGE_NAME.lower()} retry {retry + 1}/{MAX_JSON_RETRIES}: "
                f"{planned.source_id} {planned.role} {type(exc).__name__}"
            )
            time.sleep(2**retry)
    raise AssertionError("unreachable")


def execute(pricing: Path, *, resume: bool) -> None:
    require_preflight(pricing)
    if os.environ.get(AUTH_ENV) != AUTH_VALUE:
        raise RuntimeError(f"set {AUTH_ENV}={AUTH_VALUE} for Stage 2B only")
    cache = completed()
    if cache and not resume:
        raise RuntimeError(f"partial Stage {STAGE_NAME} state exists; use --resume")
    # This runner owns the only stage budget guard. Remove inherited adapter
    # settings so an earlier stage cannot create a second, stale budget ledger.
    for name in ADAPTER_BUDGET_ENV:
        os.environ.pop(name, None)
    state = current_state()
    guard = BudgetGuard(
        PriceTable.from_file(pricing), CostLedger(LEDGER),
        ceiling_usd=CEILING_USD, warning_usd=WARNING_USD,
    )
    cohort, controls, dataset, labels = inputs()
    control_contract_hash = sha(DESIGN / "genuine_change_control_plan.csv")
    for index, row in enumerate(cohort, 1):
        source = dataset[row["source_id"]]
        control = controls[row["source_id"]]
        donor = dataset[control["donor_id"]]
        dimension = row["same_intent_dimension"]
        generator_prompt = variant_prompt(source, dimension)
        generator_id = identity(
            row, provider="openai", model="gpt-4.1-mini-2025-04-14",
            role="GENERATOR", condition="GENERATION", prompt=generator_prompt,
            schema=VARIANT_SCHEMA, dimension=dimension, variant_hash="",
        )
        generated = perform_json_retry(
            cache, guard, state, generator_id,
            lambda: call_openai(
                generator_id.model, generator_prompt, "label_preserving_variant", VARIANT_SCHEMA
            ),
        )
        variant = normalize_variant(str(generated["variant_text"]))
        variant_hash = stable_hash(variant)
        verification_prompt = verifier_prompt(source, variant, labels)
        verification_schema = {
            "response_schema": VERIFY_SCHEMA,
            "generation_config": {"thinking_level": "low"},
        }
        verifier_id = identity(
            row, provider="google", model="gemini-3.5-flash", role="VERIFIER",
            condition="INDEPENDENT_VERIFICATION", prompt=verification_prompt,
            schema=verification_schema, dimension=dimension, variant_hash=variant_hash,
            parent_identity_hash=generator_id.key,
        )
        verified = perform_json_retry(
            cache, guard, state, verifier_id,
            lambda: call_google(verifier_id.model, verification_prompt, VERIFY_SCHEMA),
        )
        computed = bool(
            verified.get("label_compatible") and verified.get("bidirectional_entailment")
            and not verified.get("contradiction") and not verified.get("information_changed")
            and not verified.get("alternative_label_collision")
        )
        accepted = bool(
            verified.get("accepted") and computed and variant and variant != source["text"]
            and protected_tokens(source["text"]) == protected_tokens(variant)
        )
        roles = [("original", source["text"], stable_hash(source["text"]), INTERVENTION_HASH)]
        if accepted:
            roles.append(("variant", variant, variant_hash, INTERVENTION_HASH))
        roles.append(("control", donor["text"], control["control_text_hash"], control_contract_hash))
        for logical_role, text, role_hash, contract_hash in roles:
            for cell_id, (provider, model, condition) in CELLS.items():
                decision_prompt = (
                    openai_decision_prompt(text, labels)
                    if condition == "VANILLA"
                    else anthropic_decision_prompt(text, labels)
                )
                schema = decision_schema(labels)
                planned = identity(
                    row, provider=provider, model=model,
                    role=f"decision_{logical_role}_{cell_id}", condition=condition,
                    prompt=decision_prompt, schema=schema,
                    dimension="HUMAN_LABELED_DONOR" if logical_role == "control" else dimension,
                    variant_hash=role_hash, contract_hash=contract_hash,
                )
                perform_json_retry(
                    cache, guard, state, planned,
                    (lambda m=model, p=decision_prompt, s=schema, pr=provider:
                     call_openai(m, p, "routing_decision", s)
                     if pr == "openai" else call_anthropic(m, p, s)),
                )
        print(
            f"stage{STAGE_NAME.lower()} {index}/{EXPECTED_SOURCE_COUNT}: "
            f"{row['source_id']} accepted={accepted}"
        )
    print(f"{STAGE_TOKEN}_EXECUTION_COMPLETE")


def status() -> None:
    state = current_state()
    complete = completed()
    print(json.dumps({
        "stage": STAGE_NAME, "target_total_sources": TARGET_TOTAL_SOURCES,
        "new_sources": EXPECTED_SOURCE_COUNT,
        "maximum_base_calls": BASE_CALLS, "completed_calls": len(complete),
        "attempt_count": state["attempt_count"], "failed_attempts": state["failed_attempts"],
        "preflight_status": json.loads(PREFLIGHT.read_text()).get("status")
        if PREFLIGHT.exists() else "MISSING",
        f"stage{NEXT_STAGE.lower()}_authorized": False,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "execute", "resume", "analyze", "status"):
        modes.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--pricing", type=Path)
    args = parser.parse_args()
    if (args.preflight or args.execute or args.resume) and args.pricing is None:
        parser.error("--pricing is required")
    if args.preflight:
        preflight(args.pricing)
    elif args.execute:
        execute(args.pricing, resume=False)
    elif args.resume:
        execute(args.pricing, resume=True)
    elif args.analyze:
        import analyze_stage2b
        analyze_stage2b.main()
    else:
        status()


if __name__ == "__main__":
    main()
