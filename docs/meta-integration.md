# How Meta models are used

CV Scientist calls **Meta Model API** directly. The default model is
`muse-wonder-1.0`; `muse-spark-1.3` can use the same tools. It does not require the
OpenAI SDK and does not call an OpenAI model. Dataset annotation uses the hosted
`sam-3.1` model with the same Meta API key. No use of Muse Image, voice or
local open-weight inference is claimed.

| Layer | Code | Model / application responsibility |
|---|---|---|
| API client | [`meta.py`](../cv_scientist/meta.py) | POST `/chat/completions` at `https://api.meta.ai/v1`; Bearer authentication, timeouts and explicit provider errors |
| Tool loop | [`investigation.py`](../cv_scientist/investigation.py) | Muse chooses tools and interprets results; application executes allowed tools and checks limits |
| Visual evidence | [`agent_tools.py`](../cv_scientist/agent_tools.py) | The agent selects FP/FN records and receives bounded original-image input and box metadata |
| Structured output | Investigation / import / research schemas | Muse emits schema-constrained answers, mappings and proposals; application validates references and values |
| Dataset review | [`dataset_review.py`](../cv_scientist/dataset_review.py) | Deterministic local audit plus Muse interpretation of a bounded, explicitly counted visual sample |
| Dataset annotation | [`annotation.py`](../cv_scientist/annotation.py), [`sam.py`](../cv_scientist/sam.py) | Muse plans class prompts; SAM 3.1 runs on every image via POST `/responses`; code validates boxes and exports a separate YOLO dataset |
| Existing-format import | [`importing.py`](../cv_scientist/importing.py) | Muse identifies source fields; code re-reads the actual numbers and publishes verified local copies |
| Adaptive evaluation | [`confidence.py`](../cv_scientist/confidence.py), [`resolution.py`](../cv_scientist/resolution.py) | Muse proposes a single allowed parameter; the external adapter measures it and code decides acceptance |

Official references: [tool calling](https://dev.meta.ai/docs/tool-calling),
[structured output](https://dev.meta.ai/docs/structured-output),
[image understanding](https://dev.meta.ai/docs/image-understanding).
SAM references: [overview](https://dev.meta.ai/docs/sam/overview),
[concept prompts](https://dev.meta.ai/docs/sam/segmenting),
[wire protocol](https://github.com/meta-models/meta-sam/blob/main/protocol/sam3.md).

## Run a live investigation

```bash
source .venv/bin/activate
cv-scientist config
cv-scientist --project demo-workspace ask \
  "Compare B0 and E1; inspect missed small defects and cite the evidence. Separate observations from possible causes." \
  --agent meta --model muse-wonder-1.0 --max-steps 8 --json
```

Configure the key once using the hidden prompt. A process environment variable
`CV_SCIENTIST_API_KEY` overrides the saved key; the project and current-directory
`.env` sources also precede the global credential file. Empty explicit keys are
errors. The global file is plaintext with mode `0600`, outside the repository.
See [configuration precedence](usage.md#install-and-configure).

The app selects its model from `--model`, `META_MODEL`, workspace `agent_model`,
then the default. `.env.example` is a template, not a working credential file.
Copying its empty key without filling it overrides the saved key and causes an error.

## Boundaries

- Only selected image previews, structured context and tool outputs are sent to
  Meta. Send real data only when you are authorized to share it with that service.
- `ask` defaults to eight requests, 24 tool calls and four images. It checks a
  cumulative 100,000 API-reported-token limit after responses; this may overshoot
  by one response and is not a monetary cap. Provider failures do not become
  successful offline answers.
- A cited reference must be available to the agent. This prevents invented IDs;
  it does not establish causation or certify every sentence.
- `ask` does not launch training. Bounded `research --agent meta` runs only the
  supported evaluation protocol, with fixed inputs and explicit budgets.
- Dataset review reports viewed images versus the total. A four-image sample
  cannot certify thousands of annotations.
- `annotate-dataset` sends up to four previews to Muse and every full-resolution,
  orientation-normalized image to SAM once per planned class. Muse makes one
  planning request; SAM makes images × classes requests, sequentially, without
  automatic retries. Empty SAM output is an unverified negative, not a confirmed
  background. Malformed/incomplete responses fail the export. Only boxes are
  exported; masks are discarded without decoding. Labels require human review.
- Scripted tests exercise the application protocol, not live model intelligence.
  A demo's offline fallback must be labeled offline, not passed off as a Muse call.
