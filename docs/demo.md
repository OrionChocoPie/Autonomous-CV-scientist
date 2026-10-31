# Three-minute demo script

This is a recording plan. A public video has not been recorded or linked by this
repository. Keep the final recording within three minutes and show only the
identified hackathon contributions.

## Prepare before recording

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/create_demo.py --output demo-workspace
cv-scientist config
cv-scientist --project demo-workspace report E1 --baseline B0 --json
```

Open the returned HTML file. Keep the terminal in the repository, with no `.env`,
credential prompt, private images or account billing pages visible. Use the
synthetic workspace for every live image view. Check the live Meta call before
recording; it requires your account and may incur charges.

| Time | Show | Suggested narration |
|---|---|---|
| 0:00–0:20 | README and contribution map | “CV engineers need evidence for their next experiment. Our earlier foundation was a local experiment workspace. During the hackathon we added a tool-using Meta investigator, verifiable evaluation and recovery, import, dataset audit and visual evidence reports.” |
| 0:20–0:50 | `compare B0 E1 --json`, then report | “These are synthetic inspection fixtures. Raising confidence increases precision, but small-defect recall falls. The report exposes the trade-off and links it to the exact image records.” |
| 0:50–1:40 | Live Muse investigation command below | “Muse chooses evidence tools, reads experiment history and inspects selected misses. The application validates citations and limits requests and images. Suggestions are hypotheses, not measured outcomes.” |
| 1:40–2:10 | Returned evidence / tool trace, image overlays | “We can inspect the original image and boxes behind the answer. Unsupported references fail validation. History persists between commands.” |
| 2:10–2:40 | Case-study aggregate table and result limitations | “We tested four actual interventions; none passed every criterion. The tool retained the negative results. Training in Modal was launched by a coding agent, not autonomously by CV Scientist.” |
| 2:40–3:00 | Architecture and repository | “The new work turns a model answer into an inspectable research process. The public demo and tests run without private data. Live Muse needs a key; arbitrary training orchestration remains outside this version.” |

Commands for the recording:

```bash
source .venv/bin/activate
cv-scientist --project demo-workspace compare B0 E1 --json
cv-scientist --project demo-workspace ask \
  "Compare B0 and E1. Inspect the missed small defects. Cite the images and recommend one controlled next test without claiming its outcome." \
  --agent meta --model muse-wonder-1.0 --max-steps 8 --json
cv-scientist --project demo-workspace context --json
```

Live response times vary; the time allocations are a recording plan, not latency
claims. If the provider is unavailable, show the offline report and clearly state
that the live Meta step could not run. Do not present scripted/offline output as
live reasoning. Include a genuine Meta interaction in the final submission video
when the service is available.

Finish the submission by publishing an accessible video URL, listing every team
member, choosing one track and verifying the public repository link. The
[submission checklist](hackathon.md#before-submitting) tracks these external steps.
