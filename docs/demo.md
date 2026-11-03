# Three-minute demo: music, screen recording and captions

**No voice-over.** Use the application screen recording, background instrumental
music and short captions burned into the video. The whole story must remain
understandable when muted. This is an editing plan, not a recorded video.

- [Detailed shot list and exact bilingual captions](video-script-ru.md)
- [English subtitles — primary hackathon track](video-subtitles.en.srt)
- [Russian subtitles — alternative track](video-subtitles.ru.srt)

Select one subtitle language; do not show both at once. Each SRT has 36 cues of
five seconds, covering exactly 03:00. Do not append an extra closing sequence.

## Screen sequence

| Time | Show | Main point |
|---|---|---|
| 0:00–0:20 | Product name, earlier foundation and new contribution map | What problem this solves and what was built during the hackathon |
| 0:20–0:50 | Synthetic B0/E1 comparison, precision/recall and a missed defect | Higher precision can hide lost recall; inspect the actual error |
| 0:50–1:30 | A genuine Muse request, returned tool trace, evidence and next step | Meta investigates through tools; code validates references and resource limits |
| 1:30–1:50 | Real annotation output if available; otherwise command help with an explicit capability-only label | Muse plans prompts; SAM 3.1 exports draft YOLO labels for human review |
| 1:50–2:20 | Separate real case-study table and Fire FP 133 → 367 | Four interventions failed the full criteria; training was launched separately |
| 2:20–2:45 | Persistent context, baseline and bounded-evaluator architecture | Reuse evidence across sessions; arbitrary training is not automated |
| 2:45–3:00 | MIT, reproduction commands and final product card | Reproducible, inspectable research workflow |

## Prepare before recording

Set up credentials before recording, away from the captured screen. Reuse an
existing demo workspace; only run the generator with a new destination.

```bash
source .venv/bin/activate
# Only if this output directory does not exist:
python examples/create_demo.py --output demo-workspace
cv-scientist --project demo-workspace report E1 --baseline B0 --json
cv-scientist --project demo-workspace compare B0 E1 --json
```

Open the HTML file returned in `data.path`. Keep a visible `Synthetic fixture`
label on these images and metrics. Their predictions are authored fixtures, not
trained-model outputs. Keep private case-study aggregates in a separate segment.

Record the real Meta request separately. It requires configured access and can
incur API charges; record the actual output rather than a scripted substitute.

```bash
source .venv/bin/activate
cv-scientist --project demo-workspace ask \
  "Compare B0 and E1. Inspect the missed small defects. Cite the image evidence and recommend one controlled next test. Separate observations from hypotheses; do not claim an unmeasured improvement." \
  --agent meta --model muse-wonder-1.0 --max-steps 8 --json
cv-scientist --project demo-workspace context --json
```

Only show image inspection and tool calls that actually appear in the returned
trace. If waiting is edited out, leave a `Waiting shortened` label on the cut.
If the service is unavailable, label the fallback as offline and change the
corresponding captions. Do not pass an offline answer off as a Muse response.

For SAM, prefer a genuine completed output from authorized images. If none has
been recorded, show the implemented command with a persistent
`Implemented feature · API run not shown` label:

```bash
source .venv/bin/activate
cv-scientist annotate-dataset --help
```

## Edit and export

Use one instrumental track without vocals that you have permission to publish.
Mute recorded system audio; the final soundtrack is background music only.
Fade in over two seconds and fade out within the final three seconds.

For a 1920×1080 frame, start with captions around 40–48 px on a dark backplate;
reserve the bottom 15–20% of the frame for them. Crop and enlarge the relevant
metric or image instead of scrolling a full terminal dump. Avoid competing
paragraphs on screen. The SRT files use at most two lines of 42 characters.

Keep contribution attribution, source-data labels and execution boundaries
visible. The case study uses pseudo-label validation and one seed; it does not
establish independent-test or new-camera quality. General-purpose autonomous
training is not claimed.

Watch the full export muted and in a small window. Check readability, caption
synchronization, total duration, music rights and the absence of credentials or
private data. Then verify the public video link without signing in and complete
the [submission checklist](hackathon.md#before-submitting).
