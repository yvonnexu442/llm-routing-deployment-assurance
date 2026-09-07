# Policy-Bounded, Provenance-Aware Deployment Assurance for LLM Request Routing

Reproducibility artifact accompanying the ICOAI 2026 study by Yihua Xu, Jiani He, Ishita Chirag Talati, Yitian Qian, Youting Wang, Zhenyu Xu, and Dingyan Shang. The manuscript itself is intentionally excluded.

## Scope

This repository contains minimal frozen artifacts for cohort construction, prompt/package identities, deployment-policy configuration, lifecycle reconstruction, and sampled human validation.

## Contents

`cohorts/` contains source, eligibility, and donor manifests. `prompts/` and `metadata/` define prompt, route-set, schema, verifier, and execution identities. `policy/` contains the policy grid. `package_states/` contains the eight-cell identity manifest. `human_validation/` contains the sampled validation outputs. `renderers/` contains deterministic local reconstruction code.

Instance-specific payloads are not archived individually when they are deterministically reconstructible from frozen templates, dataset inputs, schemas, and renderers. See `prompts/RECONSTRUCTION.md` and `metadata/VERIFIER_RECONSTRUCTION.md`.

Run `python verify_artifact.py` to verify hashes and package references. Full BANKING77 and CLINC150 datasets are not redistributed; stable IDs and labels permit reconstruction from the original datasets under their licenses.

## Citation

See `CITATION.cff`. ACM publication metadata and DOI are forthcoming.

## License

Original artifact code and documentation are released under the MIT License. Third-party datasets, models, and resources remain governed by their original terms.
