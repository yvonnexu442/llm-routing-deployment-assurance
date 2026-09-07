import hashlib,json
from pathlib import Path
r=Path(__file__).parent
files=[]
for p in sorted(r.rglob('*')):
 if p.is_file() and '.git' not in p.parts and p.name not in {'ARTIFACT_MANIFEST.json','ARTIFACT_MANIFEST.md','RELEASE_CANDIDATE_REPORT.md','build_manifest.py'}:
  files.append({'relative_path':str(p.relative_to(r)),'artifact_type':'sanitized reproducibility artifact','description':'Frozen component for cohort, identity, policy, validation, or deterministic reconstruction','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_of_truth':'private source-of-truth component; logical provenance only'})
(r/'ARTIFACT_MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n')
(r/'ARTIFACT_MANIFEST.md').write_text('# Artifact manifest\n\nSee `ARTIFACT_MANIFEST.json` for SHA-256 hashes and logical provenance.\n')
