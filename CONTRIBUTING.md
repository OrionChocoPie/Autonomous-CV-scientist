# Contributing

Use a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/ -q
```

Keep changes focused and preserve existing CLI/artifact contracts. Reproduce bugs
before changing behavior; add tests for new behavior, invalid inputs, repeat
runs and missing files. Tests must run without paid API calls, private datasets
or a GPU. Integration tests should use temporary files and scripted clients.
Follow [AGENTS.md](AGENTS.md) for this repository's development rules.

To reproduce the public demo, choose a new output directory:

```bash
source .venv/bin/activate
python examples/create_demo.py --output demo-workspace
cv-scientist --project demo-workspace report E1 --baseline B0 --json
```

Keep API keys in `cv-scientist config` or the process environment. Do not include
credentials, private images, datasets, weights or `.cv-scientist` state in a pull
request. Local reports, demo workspaces and development notes are ignored by Git.
When reporting issues, include the command, sanitized error and a synthetic
reproduction. Avoid attaching raw agent reports containing private project data.

The application is licensed under MIT. Third-party packages, model weights,
services and datasets retain their own licenses and terms; a contribution must
not claim rights to third-party assets it does not include.
