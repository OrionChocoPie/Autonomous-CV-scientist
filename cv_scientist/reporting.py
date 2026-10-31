"""Self-contained local evidence reports; no model, server or network required."""
from __future__ import annotations

import hashlib
from html import escape
import json
import os
from pathlib import Path
import tempfile

from .agent_tools import AnalysisTools
from .compare import compare_runs
from .diagnose import diagnose
from .journal import local_path, redact
from .predictions import load_predictions, match_boxes, truth_box


STYLE = """
:root{font:16px/1.5 system-ui,sans-serif;color:#233144;background:#f4f6f9}
body{max-width:1180px;margin:40px auto;padding:0 24px}h1{font-size:32px;margin-bottom:8px}
h2{margin-top:0}h3{margin-bottom:8px}a{color:#175eb8;overflow-wrap:anywhere}
section,article{background:white;border:1px solid #dce2eb;border-radius:12px;padding:24px;margin:20px 0}
table{border-collapse:collapse;width:100%;min-width:520px;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e2e7ee;vertical-align:top;overflow-wrap:anywhere}
th{background:#f1f4f8;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace}
.scroll{overflow:auto}.muted{color:#657287}.warning{color:#865300;background:#fff5d8;padding:12px;border-radius:6px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:18px}
.gallery article{margin:0;min-width:0;padding:16px}.overlay{position:relative}.overlay img{display:block;width:100%;height:auto}
.overlay svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.gt{stroke:#00e49c}.prediction{stroke:#ffad33}.box{fill:none;stroke-width:2;vector-effect:non-scaling-stroke;pointer-events:stroke}
.legend{font-size:13px}.tag{display:inline-block;background:#eaf0fa;padding:3px 9px;border-radius:5px}
nav{display:flex;gap:18px;flex-wrap:wrap;margin-top:20px}details{margin-top:10px}summary{cursor:pointer}
@media print{body{margin:0;padding:0}section,article{break-inside:avoid}nav{display:none}}
"""
SCRIPT = """document.querySelectorAll('.overlay').forEach(function(layer){
 const img=layer.querySelector('img'),svg=layer.querySelector('svg');
 function size(){svg.setAttribute('viewBox','0 0 '+img.naturalWidth+' '+img.naturalHeight);}
 function failed(){layer.hidden=true;layer.nextElementSibling.hidden=false;}
 img.addEventListener('load',size);img.addEventListener('error',failed);
 if(img.complete){if(img.naturalWidth)size();else failed();}
});"""


def _text(value): return escape(str(value), quote=True)
def _anchor(identity): return "e-" + hashlib.sha256(str(identity).encode()).hexdigest()[:16]
def _json(value): return _text(json.dumps(redact(value), ensure_ascii=False, indent=2, allow_nan=False))


def _table(headers, rows):
    if not rows: return '<p class="muted">Нет данных.</p>'
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{_text(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _link(exp): return f'<a href="#{_anchor(exp.id)}">{_text(exp.id)}</a>'


def _artifact(exp, index=None):
    title = f"{exp.id}:predictions.json" + (f"#{index}" if index is not None else "")
    if not exp.run_dir: return _text(title) + ' — нет локального каталога'
    root = Path(exp.run_dir).resolve()
    path = (root / "predictions.json").resolve()
    if not path.is_relative_to(root) or not path.is_file(): return _text(title) + ' — файл недоступен'
    return f'<a href="{_text(path.as_uri())}">{_text(title)}</a>'


def _boxes(row, matched_truth, matched_preds):
    out = []
    for kind, objects, matched in (("gt", row["truth"], matched_truth), ("prediction", row["predictions"], matched_preds)):
        for i, obj in enumerate(objects):
            box = truth_box(obj) if kind == "gt" else obj["box"]
            x1, y1, x2, y2 = box
            label = obj.get("class_id", "без класса") if isinstance(obj, dict) else "без класса"
            caption = f"{kind} #{i}: {label}; " + ("TP" if i in matched else "FN" if kind == "gt" else "FP")
            out.append(f'<rect class="box {kind}" x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}"><title>{_text(caption)}</title></rect>')
    return ''.join(out)


def _gallery(workspace, exp, records, remaining):
    cards, used = [], 0
    tools = AnalysisTools(workspace)
    def errors(item):
        _, row = item
        t, p, _ = match_boxes(row, exp.config.get("iou_match", 0.5))
        return (-(len(row["truth"]) + len(row["predictions"]) - len(t) - len(p)), item[0])
    ranked = sorted(enumerate(records), key=errors)
    for index, row in ranked[:remaining]:
        t, p, _ = match_boxes(row, exp.config.get("iou_match", 0.5))
        counts = f"TP {len(p)} · FP {len(row['predictions'])-len(p)} · FN {len(row['truth'])-len(t)}"
        try:
            result = tools.view_error(exp.id, row["image"])
            source = result.images[0]["image_url"]["url"]
            visual = f'<div class="overlay"><img src="{source}" alt="{_text(row["image"])}"><svg xmlns="http://www.w3.org/2000/svg" aria-label="GT and predicted boxes">{_boxes(row, t, p)}</svg></div><p class="warning" hidden>Изображение повреждено или не поддерживается браузером.</p>'
        except (OSError, ValueError) as error:
            visual = f'<p class="warning">Изображение недоступно: {_text(error)}</p>'
        cards.append(f'<article id="{_anchor(exp.id + ":predictions.json#" + str(index))}"><h3>{_text(row["image"])}</h3><p>{_text(counts)}</p>{visual}'
            f'<p class="legend">GT — зелёный · Prediction — оранжевый. Статус и класс — в подсказке бокса.</p><p>{_artifact(exp,index)}</p>'
            f'<details><summary>Координаты, метки и исходные counts</summary><pre>{_json(row)}</pre></details></article>')
        used += 1
    if len(records) > used: cards.append(f'<p class="muted">Показано {used} из {len(records)} изображений; полный список доступен в артефакте.</p>')
    return '<div class="gallery">' + ''.join(cards) + '</div>', used


def _experiment(workspace, exp, remaining):
    diagnosis = diagnose(exp)
    records = load_predictions(exp, optional=True)
    findings = []
    for finding in diagnosis.findings:
        refs = []
        for ref in finding.evidence:
            if ref["kind"] == "artifact":
                suffix = ref["id"].rsplit("#", 1)[-1]
                refs.append(_artifact(exp, int(suffix)) if suffix.isdigit() else _artifact(exp))
            else: refs.append(_link(exp))
        findings.append([_text(finding.severity), _text(finding.title), _text(finding.value), '<br>'.join(refs)])
    lineage, current, seen = [], exp, set()
    while current:
        if current.id in seen:
            lineage.append('Цикл в ancestry'); break
        seen.add(current.id); lineage.append(_text(current.id))
        if not current.parent_id: break
        parent = workspace.memory.get_experiment(current.parent_id)
        if parent is None: lineage.append(_text(current.parent_id) + ' (не импортирован)')
        current = parent
    matrix = diagnosis.classification
    confusion = _table(["GT", "Prediction", "Объекты"], [[_text(cell["truth"] if cell["truth"] is not None else "∅ фон"),
        _text(cell["prediction"] if cell["prediction"] is not None else "∅ пропуск"), str(cell["count"])] for cell in matrix["matrix"]])
    gallery, used = _gallery(workspace, exp, records, remaining)
    limitation = ''.join(f'<p class="warning">{_text(item)}</p>' for item in matrix["limitations"])
    if not records.available: limitation += '<p class="warning">predictions.json недоступен. Диагностика основана на сохранённых метриках.</p>'
    html = f'<section id="{_anchor(exp.id)}"><h2>{_text(exp.id)} · {_text(exp.name)}</h2><p>Ancestry: {" ← ".join(lineage)}</p>'
    html += '<h3>Измеренные метрики</h3>' + _table(["Метрика", "Значение", "Источник"], [[_text(k), _text(v), _link(exp)] for k,v in sorted(exp.metrics.items())])
    html += f'<details><summary>Конфигурация и происхождение</summary><pre>{_json(exp.to_dict())}</pre></details>'
    html += '<h3>Диагностика</h3>' + _table(["Severity", "Наблюдение", "Значение", "Evidence"], findings)
    html += f'<h3>Confusion matrix</h3><p>Matching: {_text(matrix["mode"])}. {_artifact(exp)}</p>{limitation}{confusion}'
    html += '<h3>Изображения и боксы</h3>' + gallery + '</section>'
    return html, used


def _comparison(a, b):
    comparison = compare_runs(a, b)
    a_config, b_config = redact(a.config), redact(b.config)
    rows = []
    for scope, table in (("overall", {"overall": comparison.overall}), ("class", comparison.per_class), ("slice", comparison.per_slice)):
        for name, metrics in sorted(table.items()):
            for key, delta in sorted(metrics.items()):
                rows.append([_text("overall" if scope == "overall" else scope + ":" + name), _text(key), _text(delta["a"]), _text(delta["b"]), f'{delta["abs"]:+.6g}'])
    configs = [[_text(key), f'<pre>{_json(a_config.get(key))}</pre>', f'<pre>{_json(b_config.get(key))}</pre>']
               for key in sorted(set(a.config) | set(b.config)) if a.config.get(key) != b.config.get(key)]
    regressions = [[_text(r["slice"]), _text(r["metric"]), f'{r["relative_delta"]:+.1%}', _link(a) + ' → ' + _link(b)] for r in comparison.regressions]
    return '<section id="comparison"><h2>Сравнение ' + _link(a) + ' → ' + _link(b) + '</h2>' + _table(
        ["Область", "Метрика", "Baseline", "Current", "Δ"], rows) + '<h3>Изменения конфигурации</h3>' + _table(
        ["Параметр", "Baseline", "Current"], configs) + '<h3>Регрессии</h3>' + _table(["Область", "Метрика", "Δ относительно baseline", "Evidence"], regressions) + '</section>'


def generate_report(workspace, experiment_id=None, baseline_id=None, research_path=None, max_examples=12):
    if type(max_examples) is not int or not 1 <= max_examples <= 24: raise ValueError("max_examples must be between 1 and 24")
    for identity in (experiment_id, baseline_id):
        if identity is not None and (not isinstance(identity, str) or not identity.strip() or len(identity) > 2048):
            raise ValueError("Experiment and baseline IDs must be nonempty strings")
    journal, research = None, ''
    if research_path is not None:
        if not isinstance(research_path, (str, Path)) or not str(research_path).strip(): raise ValueError("Invalid research path")
        path = Path(research_path)
        if path.is_symlink() or path.resolve().parent != (Path(workspace.dir) / "reports").resolve():
            raise ValueError("Research report must belong to this workspace")
        journal = json.loads(path.read_text())
        if not isinstance(journal, dict) or journal.get("journal_version") != 2: raise ValueError("Invalid research journal")
        if (any(not isinstance(journal.get(key), str) or not journal[key].strip() for key in ("baseline_id", "goal", "status"))
                or any(not isinstance(journal.get(key, []), list) or any(not isinstance(row, dict)
                    or not isinstance(row.get("experiment_id"), str) or not row["experiment_id"].strip() for row in journal.get(key, []))
                    for key in ("history", "attempts"))):
            raise ValueError("Invalid research journal fields")
        for key in ("selected_run", "best_feasible_run"):
            if journal.get(key) is not None and (not isinstance(journal[key], str) or not journal[key].strip()):
                raise ValueError("Invalid research selection")
        if baseline_id and baseline_id != journal["baseline_id"]: raise ValueError("Baseline differs from the research journal")
        baseline_id = journal["baseline_id"]
        history = journal.get("history", [])
        experiment_id = experiment_id or journal.get("selected_run") or (history[-1]["experiment_id"] if history else baseline_id)
        research = f'<section id="research"><h2>Исследование</h2><p>{_text(journal["goal"])}</p><p>Статус: {_text(journal["status"])} · Параметр: {_text(journal.get("parameter","confidence"))}</p>'
        research += f'<p>{_text(journal.get("reason",journal.get("error","")))}</p><p><a href="{_text(path.resolve().as_uri())}">Исходный журнал исследования</a></p>'
        research += '<pre>' + _json({k: journal.get(k) for k in ("criteria", "selected_run", "best_feasible_run", "experiment_budget", "usage", "attempts")}) + '</pre></section>'
    exp = workspace.memory.get_experiment(experiment_id) if experiment_id else workspace.memory.latest_experiment()
    if experiment_id and exp is None: raise ValueError("Unknown experiment: " + experiment_id)
    base = workspace.memory.get_experiment(baseline_id) if baseline_id else workspace.memory.working_baseline()
    if baseline_id and base is None: raise ValueError("Unknown baseline: " + baseline_id)
    experiments = [exp] if exp else []
    if base and all(e.id != base.id for e in experiments): experiments.append(base)
    if journal:
        for item in journal.get("history", []):
            known = workspace.memory.get_experiment(item["experiment_id"])
            if known and all(e.id != known.id for e in experiments): experiments.append(known)
    content = research
    if base and exp and base.id != exp.id: content += _comparison(base, exp)
    remaining = max_examples
    for experiment in experiments:
        section, used = _experiment(workspace, experiment, remaining)
        content += section; remaining -= used
    if not experiments: content += '<section><p>Нет импортированных экспериментов.</p></section>'
    identity = json.dumps([experiment_id or (exp.id if exp else None), baseline_id or (base.id if base else None), str(research_path or "")])
    path = local_path(workspace, "reports/report-" + hashlib.sha256(identity.encode()).hexdigest()[:16] + ".html")
    navigation = ''.join(f'<a href="#{_anchor(e.id)}">{_text(e.id)}</a>' for e in experiments)
    html = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'">'
    html += f'<title>CV Scientist · {_text(workspace.name)}</title><style>{STYLE}</style><body><h1>CV Scientist · {_text(workspace.name)}</h1>'
    html += '<p class="muted">Локальный снимок измерений и evidence. Причинные объяснения требуют отдельной проверки.</p><nav>' + navigation + '</nav>' + content + f'<script>{SCRIPT}</script></body></html>'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name); file.write(html); file.flush(); os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    workspace.record_event("report", "completed", report_path=str(path), experiments=len(experiments))
    return path
