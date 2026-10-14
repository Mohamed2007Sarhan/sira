"""Write sira/manifest.json (sha256 of every package file). Run before building a release; users can run: python -m sira --verify-install"""
import hashlib, json, sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent / "sira"
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
(root / "manifest.json").write_text(json.dumps({"version": "2.0.0", "files": files}, indent=1))
print(f"manifest with {len(files)} files written")
