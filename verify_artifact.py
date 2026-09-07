import hashlib, json
import csv
from pathlib import Path
root=Path(__file__).parent
items=json.loads((root/'ARTIFACT_MANIFEST.json').read_text())
errors=[]
for item in items:
 p=root/item['relative_path']
 if not p.exists(): errors.append(f'missing: {item["relative_path"]}'); continue
 got=hashlib.sha256(p.read_bytes()).hexdigest()
 if got != item['sha256']: errors.append(f'hash mismatch: {item["relative_path"]}')
state=root/'package_states/package_state_manifest.json'
if not state.exists(): errors.append('missing package state manifest')
csv_path=root/'package_states/package_state_manifest.csv'
if not csv_path.exists(): errors.append('missing package-state CSV')
else:
 with csv_path.open(newline='') as f:
  rows=list(csv.DictReader(f))
 if len(rows)!=8: errors.append(f'expected 8 package cells, found {len(rows)}')
 for row in rows:
  for field in ('prompt_template_artifact','route_set_artifact','schema_artifact','verifier_artifact','execution_config_artifact','policy_artifact','deterministic_renderer_source'):
   p=root/row[field]
   if not p.exists(): errors.append(f'missing package component: {row[field]}')
   else:
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got != row[field.replace('_artifact','_sha256').replace('deterministic_renderer_source','deterministic_renderer_sha256')]: errors.append(f'package hash mismatch: {row[field]}')
if errors:
 print('\n'.join(errors)); raise SystemExit(1)
print(f'OK: {len(items)} frozen artifacts verified')
