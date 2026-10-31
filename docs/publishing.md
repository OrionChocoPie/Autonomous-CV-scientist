# Publish this repository

The prepared commit contains application source, tests, synthetic-demo generator,
CI, documentation and the MIT license. Git ignores credentials, local research
state, generated workspaces, reports, datasets, weights and internal development
notes. Existing ignored files are retained locally.

Before upload:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
git status --short
git ls-files
```

Review the tracked list, including history, for private material. Ignore rules do
not remove a previously committed secret. The preparation scan checks known
credential patterns and matches against local keys without printing their values;
it is not a guarantee that every possible secret format has been recognized.

## GitHub handoff

Create an empty **public** repository in your chosen account. With GitHub CLI,
the following commands are an example of the publishing step:

```bash
gh repo create cv-scientist --public --source . --remote origin
git push --set-upstream origin HEAD
```

They are not executed by repository preparation. If you already created a GitHub
repository, add its actual remote instead of creating another one. Verify the
public README in a signed-out browser and check the Actions result. No hosted-CI
success is claimed until the workflow actually runs on GitHub.

Suggested repository description:

> Evidence-driven computer vision research with Meta Muse: experiment memory,
> class-aware error analysis and bounded evaluation.

Suggested topics: `computer-vision`, `meta-ai`, `muse`, `research-tools`,
`experiment-tracking`, `object-detection`.

The repository is one submission artifact. Record and publish the three-minute
video, list all contributors and submit the links using the organizer's form.
See [the event checklist](hackathon.md).
