"""Apply explicitly reviewed duplicate removals to a staged train copy."""
import hashlib
import json
from pathlib import Path

from cv_scientist.predictions import iou
from scripts.threats_artifacts import file_sha256, yolo_truth


def patch_identity(patch):
    raw = json.dumps(patch, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "threats-train-reviewed-" + hashlib.sha256(raw).hexdigest()[:16]


def apply_label_patch(root, patch, parent_version):
    if (not isinstance(patch, dict) or patch.get("parent_dataset_version") != parent_version
            or patch.get("policy") != "separate_visible_fragments"
            or patch.get("reviewer") != "agent_visual_review"):
        raise ValueError("Invalid reviewed patch provenance or instance policy")
    changes = patch.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("Reviewed patch must contain changes")
    prepared, seen, cases = [], set(), set()
    for change in changes:
        relative = Path(change["label"])
        if (relative.is_absolute() or relative.parts[:3] != ("yolo", "labels", "train")
                or len(relative.parts) != 4 or relative.suffix != ".txt" or ".." in relative.parts
                or relative.as_posix() in seen):
            raise ValueError("Patch may change each train label once only")
        seen.add(relative.as_posix())
        target = Path(root) / relative
        if file_sha256(target) != change["original_sha256"]:
            raise ValueError("Original train label checksum mismatch")
        lines = target.read_text().splitlines()
        truth = yolo_truth("\n".join(lines), 1, 1)
        removals, keeps = set(), set()
        for review in change["reviews"]:
            if (review.get("decision") != "delete_duplicate" or not review.get("note")
                    or not isinstance(review.get("case_id"), str) or review["case_id"] in cases):
                raise ValueError("Each removal needs a unique explicit visual review")
            cases.add(review["case_id"])
            remove, keep = review["remove_index"], review["keep_index"]
            if (type(remove) is not int or type(keep) is not int or remove == keep
                    or not 0 <= min(remove, keep) or max(remove, keep) >= len(truth)
                    or remove in removals):
                raise ValueError("Invalid reviewed line indices")
            a, b = truth[remove], truth[keep]
            if (a["class_id"] not in (0, 1) or a["class_id"] != b["class_id"]
                    or iou(a["box"], b["box"]) < .9):
                raise ValueError("Reviewed removal does not match the screened fire/smoke pair")
            removals.add(remove)
            keeps.add(keep)
        if not removals or removals & keeps:
            raise ValueError("Each removed box must retain its reviewed counterpart")
        prepared.append((target, "\n".join(line for index, line in enumerate(lines) if index not in removals) + "\n"))
    if len(cases) < 10:
        raise ValueError("At least 10 confirmed annotation errors are required for this training fork")
    # Validate the entire patch before changing any staged file.
    for target, content in prepared:
        temporary = target.with_suffix(".reviewed.tmp")
        temporary.write_text(content)
        temporary.replace(target)
    return {"dataset_version": patch_identity(patch), "changed_train_labels": len(prepared),
            "removed_duplicate_boxes": len(cases), "val_test_changed": False}
