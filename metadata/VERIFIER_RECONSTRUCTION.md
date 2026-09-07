# Deterministic verifier reconstruction

The verifier model, temperature, retry limit, fail-closed behavior, schema source, and prompt template are frozen in `verifier_identity.json`, `verifier_prompt_template.txt`, and the renderer copy. A verifier payload is rendered from the source/rewrite record, expected label, ordered alternative labels, and the frozen verifier template; the structured output is checked against the frozen schema and eligibility rule. No provider call is required to reconstruct the payload.
