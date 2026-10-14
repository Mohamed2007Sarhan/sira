"""Write sira/manifest.json (sha256 of every package file). Run before a release; users run: python -m sira --verify-install"""
import hashlib, json, re
from pathlib import Path
root = Path(__file__).resolve().parent.parent / "sira"
version = re.search(r'__version__\s*=\s*"([^"]+)"', (root / "__init__.py").read_text()).group(1)
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
(root / "manifest.json").write_text(json.dumps({"version": version, "files": files}, indent=1))
print(f"manifest v{version} with {len(files)} files written")
