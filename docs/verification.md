# Verification record

## Local checks

The preparation checks used the repository's `.venv` on Python 3.14.

| Check | Observed result |
|---|---|
| Editable installation | `python -m pip install -e '.[dev]'` completed |
| Full test suite | `512 passed in 17.49s` |
| Public demo | Generated five synthetic images, imported B0/E1, compared and rendered an HTML report without a model or network |
| Expected trade-off | B0 → E1 recall delta = −0.25; the report includes embedded images and evidence pointers |
| Offline guardrail evaluation | 16/16 scenarios passed; zero paid API calls; scripted clients, not a reasoning-quality benchmark |
| Retrieval fixture | 12/12 cases passed; mean recall@3 = 1.0 on this fixed fixture |
| Wheel build | `python -m pip wheel . --no-deps` completed; 35 core modules and license included, no private state or weights |
| Documentation | 35 local Markdown links checked; no missing targets at that check |
| GitHub workflow | YAML parsed, Python 3.10/3.12/3.14 matrix and read-only repository permissions checked |
| Credential scan | No known token patterns or exact matches against local credentials found in publication candidates or the existing history |
| Whitespace | `git diff --check` completed |

The checksum helper was also tested without `hashlib.file_digest`, an API absent
from Python 3.10. Empty, single-byte and multi-block files produce the expected
SHA-256 using the compatible streaming implementation. This simulation does not
replace running the whole suite on Python 3.10.

## Boundaries

Hosted GitHub Actions has not run during local preparation. The workflow's other
Python versions remain unverified until their jobs finish. No live Meta API call
was made during repository preparation; existing live usage and the private
case-study outputs are separate from these offline checks. No public repository,
video, submission-form completion or production deployment is claimed here.

The secret scan covers recognized formats and known local values, not every
possible credential representation. Original ignored local files were retained.
Raw research state, private data and development logs are not publication assets.
