"""
Flask web interface for AI Handwriter.

Separate pages, deliberately kept separate per the PRD's own workflow (and per
explicit request): training a personal handwriting profile is a distinct,
slower, occasional task from writing a document and exporting G-code, which
should stay fast.

  /            Write & Export  — paste text, pick a page/printer profile,
               preview the handwriting, download SVG/PNG/G-code.
  /train       Personalize     — capture-tool instructions, upload samples,
               fine-tune a profile, save it for use on the main page.
               Returns a background job id; both pages poll /jobs/<job_id>
               for live progress without holding the submission request open.
  /profiles    Profiles        — view, rename, and delete saved profiles.

This file only handles HTTP glue (forms, file IO, calling the pipeline).
All the actual work happens in the sibling modules — see README.md for what
each one does.
"""
import glob
import json
import os
import time
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
from datetime import datetime, timezone

from flask import Flask, render_template, request, send_file, url_for, jsonify, redirect

from ai.logging_utils import configure_logging
from ai.training.validation import save_validated_profile
from ai.profiles import ordered_profiles, save_order, move_profile, describe_profile, SIDECARS
from ai.inference.runtime import create_model, resolve_device, normalize_device as normalize_generation_device
from ai.training.samples import load_profile_alphabet
from ai.training.style import load_style

configure_logging()
logger = logging.getLogger('ai.web')

from ai.inference.hw_model import HandwritingModel, ALPHABET, sanitize_line
from document.calibration import PageTemplate, PrinterProfile
from document.layout import generate_document
from document.render import document_to_svg, document_to_png
from document.gcode import generate_gcode, validate_document, calibration_pattern_gcode, GCodeValidationError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(BASE_DIR, "webapp")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
PREVIEWS_DIR = os.path.join(WEBAPP_DIR, "static", "previews")
DOWNLOADS_DIR = os.path.join(WEBAPP_DIR, "static", "downloads")
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(PREVIEWS_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(WEBAPP_DIR, "templates"),
            static_folder=os.path.join(WEBAPP_DIR, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

_MODEL_CACHE = {}  # (profile_name, resolved_device) -> inference model
_JOBS = {}
_JOB_LOCK = threading.Lock()
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="handwriter")
# shortcut: jobs live in this local process; use durable storage for multi-worker hosting.


def _start_job(kind, work):
    with _JOB_LOCK:
        if any(job["status"] not in ("done", "error") for job in _JOBS.values()):
            raise ValueError("Another job is running. Wait for it to finish.")
        while len(_JOBS) >= MAX_OUTPUT_JOBS:
            del _JOBS[next(iter(_JOBS))]
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {"status": "queued", "kind": kind, "progress": 0}
    def run():
        started = time.monotonic()
        def update(**values):
            with _JOB_LOCK:
                _JOBS[job_id].update(values, elapsed_s=round(time.monotonic() - started, 1))
        try:
            update(status="starting")
            logger.debug('Job started job=%s kind=%s', job_id, kind)
            result = work(job_id, update)
            update(status="done", progress=100, result=result)
            logger.debug('Job finished job=%s kind=%s elapsed_s=%.1f',
                         job_id, kind, time.monotonic() - started)
        except Exception as error:
            logger.exception('Job failed job=%s kind=%s elapsed_s=%.1f',
                             job_id, kind, time.monotonic() - started)
            update(status="error", error=str(error))
    _WORKER.submit(run)
    return job_id

# Preview/G-code files pile up in static/ over time (every generation makes
# new ones and nothing ever deleted them). Purge anything older than this,
# checked once per write-page request — cheap, and keeps disk from growing
# unbounded on a machine left running for a while.
MAX_OUTPUT_AGE_S = 2 * 60 * 60  # 2 hours
MAX_OUTPUT_JOBS = 50            # also cap by count regardless of age


def _cleanup_old_outputs():
    now = time.time()
    job_dirs = sorted(
        (p for p in glob.glob(os.path.join(PREVIEWS_DIR, "*")) if os.path.isdir(p)),
        key=os.path.getmtime,
    )
    for i, d in enumerate(job_dirs):
        too_old = (now - os.path.getmtime(d)) > MAX_OUTPUT_AGE_S
        too_many = (len(job_dirs) - i) > MAX_OUTPUT_JOBS
        if too_old or too_many:
            for f in glob.glob(os.path.join(d, "*")):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(d)
            except OSError:
                pass
    for f in glob.glob(os.path.join(DOWNLOADS_DIR, "*.gcode")):
        if (now - os.path.getmtime(f)) > MAX_OUTPUT_AGE_S:
            try:
                os.remove(f)
            except OSError:
                pass


def list_profiles():
    return ordered_profiles(PROFILES_DIR)


def _invalidate_model(name):
    for key in list(_MODEL_CACHE):
        if key == name or (isinstance(key, tuple) and key[0] == name):
            del _MODEL_CACHE[key]


def validate_profile_name(name):
    if not name or len(name) > 64 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in name):
        raise ValueError("Profile name must use 1–64 letters, numbers, hyphens, or underscores.")


@app.route("/profiles", methods=["GET", "POST"])
def profiles_page():
    error = None
    if request.method == "POST":
        try:
            with _JOB_LOCK:
                if any(job["status"] not in ("done", "error") for job in _JOBS.values()):
                    raise ValueError("Wait for the running job to finish before changing profiles.")
                name = request.form.get("profile", "")
                if name not in list_profiles():
                    raise ValueError("Choose an existing handwriting profile.")
                if name == "default":
                    raise ValueError("The default profile cannot be renamed or deleted.")
                weights = os.path.join(PROFILES_DIR, f"{name}.npz")
                action = request.form.get("action")
                previous_order = list_profiles()
                has_order = os.path.exists(os.path.join(PROFILES_DIR, '.order.json'))
                if action == "rename":
                    new_name = request.form.get("profile_name", "").strip()
                    validate_profile_name(new_name)
                    if new_name == "default" or new_name in list_profiles():
                        raise ValueError("Choose a new name that is not already in use.")
                    new_weights = os.path.join(PROFILES_DIR, f"{new_name}.npz")
                    pairs = [(os.path.join(PROFILES_DIR, name + suffix),
                              os.path.join(PROFILES_DIR, new_name + suffix)) for suffix in SIDECARS]
                    if any(os.path.exists(target) for _, target in pairs):
                        raise ValueError("That name already has saved profile data. Choose another name.")
                    moved = []
                    try:
                        for source, target in pairs:
                            if os.path.exists(source):
                                os.rename(source, target)
                                moved.append((source, target))
                        os.rename(weights, new_weights)
                    except OSError:
                        for source, target in reversed(moved):
                            os.rename(target, source)
                        raise
                    if has_order:
                        save_order(PROFILES_DIR, [new_name if p == name else p for p in previous_order])
                elif action == "delete":
                    os.remove(weights)
                    for suffix in SIDECARS:
                        sidecar = os.path.join(PROFILES_DIR, name + suffix)
                        if os.path.exists(sidecar):
                            os.remove(sidecar)
                    if has_order:
                        save_order(PROFILES_DIR, [p for p in previous_order if p != name])
                elif action in ('move_up', 'move_down'):
                    move_profile(PROFILES_DIR, name, -1 if action == 'move_up' else 1)
                else:
                    raise ValueError("Choose Rename, Delete, or a reorder direction.")
                _invalidate_model(name)
            return redirect(url_for("profiles_page"))
        except ValueError as exc:
            error = str(exc)
        except OSError:
            error = "Could not update the profile files. Check file permissions and refresh the profile list before trying again."
    names = list_profiles()
    details = {name: describe_profile(PROFILES_DIR, name) for name in names}
    return render_template("profiles.html", profiles=names, details=details, error=error), 400 if error else 200


def get_model(profile_name: str, device='auto') -> HandwritingModel:
    if profile_name not in list_profiles():
        raise ValueError("Choose an existing handwriting profile.")
    device = resolve_device(device)
    key = (profile_name, device)
    if key not in _MODEL_CACHE:
        weights_path = os.path.join(PROFILES_DIR, f"{profile_name}.npz")
        alphabet = load_profile_alphabet(weights_path)
        model = create_model(weights_path, alphabet, device=device)
        model.style_samples = load_style(weights_path, alphabet)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def form_float(name, default, low=None, high=None):
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name.replace('_', ' ')} must be a number.")
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name.replace('_', ' ')} is outside its allowed range.")
    return value


def form_int(name, default, low=None, high=None):
    value = form_float(name, default, low, high)
    if not value.is_integer():
        raise ValueError(f"{name.replace('_', ' ')} must be a whole number.")
    return int(value)


DEFAULT_FORM = {
    "text": "Dear team,\n\nThis is a sample letter written by the AI Handwriter.\n\nSincerely,\nYou",
    "profile": "default", "page_width_mm": 210, "page_height_mm": 297,
    "margin_left_mm": 20, "margin_right_mm": 15, "margin_top_mm": 20, "margin_bottom_mm": 20,
    "lined": "on", "line_spacing_mm": 9, "paragraph_spacing_mm": 4,
    "handwriting_size_mm": 6, "bias": 1.5, "seed": 7,
    "chunk_chars": 20, "attempts": 3, "steps_per_char": 50,
    "letter_spacing": 1, "word_spacing": 1, "slant": 0,
    "generation_device": "auto",
    "bed_width_mm": 220, "bed_height_mm": 220,
    "paper_origin_x_mm": 10, "paper_origin_y_mm": 10,
    "pen_up_z_mm": 5, "pen_down_z_mm": 0,
    "writing_speed_mm_s": 60, "travel_speed_mm_s": 150,
}


@app.route("/", methods=["GET"])
def write_page():
    return render_template("index.html", profiles=list_profiles(), form=DEFAULT_FORM, result=None)


@app.route("/generate", methods=["POST"])
def generate():
    try:
        text = request.form.get("text", "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text or len(text) > 20000:
            raise ValueError("Enter between 1 and 20,000 characters.")
        profile = request.form.get("profile", "default")
        if profile not in list_profiles():
            raise ValueError("Choose an existing handwriting profile.")
        device = normalize_generation_device(request.form.get('generation_device', 'auto'))
        page = PageTemplate(
            width_mm=form_float("page_width_mm", 210, 20, 1000),
            height_mm=form_float("page_height_mm", 297, 20, 1000),
            **{key: form_float(key, DEFAULT_FORM[key], 0, 500) for key in
               ("margin_left_mm", "margin_right_mm", "margin_top_mm", "margin_bottom_mm")},
            lined=request.form.get("lined") == "on",
            line_spacing_mm=form_float("line_spacing_mm", 9, 2, 50),
            paragraph_spacing_mm=form_float("paragraph_spacing_mm", 4, 0, 50),
            handwriting_size_mm=form_float("handwriting_size_mm", 6, 1, 20),
        )
        if page.text_width_mm < 10 or page.text_height_mm < page.line_spacing_mm + page.handwriting_size_mm:
            raise ValueError("Margins must leave enough room for at least one line.")
        printer = PrinterProfile(
            bed_width_mm=form_float("bed_width_mm", 220, 20, 2000),
            bed_height_mm=form_float("bed_height_mm", 220, 20, 2000),
            usable_x_max=form_float("bed_width_mm", 220, 20, 2000),
            usable_y_max=form_float("bed_height_mm", 220, 20, 2000),
            paper_origin_x_mm=form_float("paper_origin_x_mm", 10, 0, 2000),
            paper_origin_y_mm=form_float("paper_origin_y_mm", 10, 0, 2000),
            pen_up_z_mm=form_float("pen_up_z_mm", 5, 0, 500),
            pen_down_z_mm=form_float("pen_down_z_mm", 0, 0, 500),
            writing_speed_mm_s=form_float("writing_speed_mm_s", 60, 1, 300),
            travel_speed_mm_s=form_float("travel_speed_mm_s", 150, 1, 500),
        )
        options = dict(
            bias=form_float("bias", 1.5, 0, 3), seed=form_int("seed", 7, 0, 2**31 - 1),
            chunk_chars=form_int("chunk_chars", 20, 12, 48), attempts=form_int("attempts", 3, 1, 5),
            steps_per_char=form_int("steps_per_char", 50, 20, 100),
            letter_spacing=form_float("letter_spacing", 1, 0.5, 1.5),
            word_spacing=form_float("word_spacing", 1, 0.5, 2), slant=form_float("slant", 0, -20, 20),
        )
        def work(job_id, update):
            _cleanup_old_outputs()
            model = get_model(profile, device=device)
            update(device=getattr(model, 'device', 'cpu'), device_name=getattr(model, 'device_name', 'CPU'))
            style_samples = getattr(model, 'style_samples', [])
            prime_strokes, prime_text = style_samples[0] if style_samples else (None, None)
            normalized = "\n".join(sanitize_line(line, set(model.alphabet)) for line in text.splitlines())
            warnings = []
            if normalized != text:
                warnings.append("Unsupported characters were replaced or removed. Check the normalized text below.")
            if not normalized.strip():
                raise ValueError("This profile does not support any characters in your text.")
            started = time.monotonic()
            def progress(done, total):
                elapsed = time.monotonic() - started
                update(status="generating", progress=round(done / total * 90), completed=done, total=total,
                       eta_s=round(elapsed / done * (total - done), 1))
            doc = generate_document(normalized, model, page, progress_cb=progress,
                                    prime_strokes=prime_strokes, prime_text=prime_text, **options)
            update(status="rendering", progress=92)
            directory = os.path.join(PREVIEWS_DIR, job_id)
            os.makedirs(directory, exist_ok=True)
            svg_paths = document_to_svg(doc, page, os.path.join(directory, "page"))
            png_paths = document_to_png(doc, page, os.path.join(directory, "page"))
            issues = validate_document(doc, page, printer)
            gcode_url, gcode_error = None, None
            try:
                gc = generate_gcode(doc, page, printer, raise_on_error=True)
                with open(os.path.join(DOWNLOADS_DIR, f"{job_id}.gcode"), "w", encoding="utf-8") as stream:
                    stream.write(gc)
                gcode_url = f"/static/downloads/{job_id}.gcode"
            except GCodeValidationError as error:
                gcode_error = str(error)
            return {"job_id": job_id, "num_pages": len(doc.pages), "gen_seconds": round(time.monotonic() - started, 1),
                    "preview_urls": [f"/static/previews/{job_id}/{os.path.basename(path)}" for path in svg_paths],
                    "svg_urls": [f"/static/previews/{job_id}/{os.path.basename(path)}" for path in svg_paths],
                    "png_urls": [f"/static/previews/{job_id}/{os.path.basename(path)}" for path in png_paths],
                    "gcode_url": gcode_url, "gcode_error": gcode_error,
                    "margin_warnings": sum(i.kind == "outside_margin" for i in issues),
                    "warnings": warnings, "normalized_text": normalized,
                    "generation_device": getattr(model, 'device', 'cpu'),
                    "generation_device_name": getattr(model, 'device_name', 'CPU'),
                    "style_primed": bool(style_samples)}
        job_id = _start_job("generation", work)
        return jsonify(job_id=job_id, progress_url=f"/jobs/{job_id}"), 202
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.route("/jobs/<job_id>")
@app.route("/train/progress/<job_id>")
def job_progress(job_id):
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return jsonify(error="Job expired or server restarted. Start again."), 404
        return jsonify(dict(job))


@app.route("/train")
def train_page():
    return render_template("train.html", profiles=list_profiles())


@app.route("/train/check-alphabet", methods=["POST"])
def train_check_alphabet():
    try:
        from ai.training.samples import build_alphabet, load_training_samples, load_profile_alphabet
        profile = request.form.get("base_profile", "default")
        if profile not in list_profiles():
            raise ValueError("Choose an existing base profile.")
        upload = request.files.get("samples_json")
        if not upload:
            raise ValueError("Choose a samples JSON file.")
        samples = load_training_samples(upload.stream)
        if not samples:
            raise ValueError("No usable samples found. Capture at least one complete phrase.")
        alphabet = build_alphabet(request.form.get("extra_chars", ""), load_profile_alphabet(os.path.join(PROFILES_DIR, f"{profile}.npz")), samples)
        return jsonify(missing_chars=[], num_samples=len(samples), alphabet_size=len(alphabet))
    except (ValueError, KeyError, TypeError) as error:
        return jsonify(error=str(error)), 400


@app.route("/train/run", methods=["POST"])
def train_run():
    try:
        from ai.training.samples import load_training_samples, build_alphabet, load_profile_alphabet
        from ai.training.devices import normalize_device
        upload = request.files.get("samples_json")
        if not upload or not upload.filename:
            raise ValueError("Choose a samples JSON file exported from Capture samples.")
        profile_name = (request.form.get("profile_name") or "").strip()
        validate_profile_name(profile_name)
        if profile_name == "default" or profile_name in list_profiles():
            raise ValueError("Choose a new profile name to preserve your existing profiles.")
        base_profile = request.form.get("base_profile", "default")
        if base_profile not in list_profiles():
            raise ValueError("Choose an existing base profile.")
        epochs = form_int("epochs", 10, 1, 40)
        lr = form_float("learning_rate", 0.00001, 0.000001, 0.001)
        device = normalize_device(request.form.get("device", "auto"))
        base_weights = os.path.join(PROFILES_DIR, f"{base_profile}.npz")
        # Parse and validate before constructing TensorFlow or starting a job.
        samples = load_training_samples(upload.stream)
        if not samples:
            raise ValueError("No usable samples found. Capture at least one complete phrase.")
        if len(samples) * epochs > 800:
            raise ValueError("Web training allows up to 800 sample-epochs. Reduce epochs or use python -m ai.training.finetune_cli.")
        base_alphabet = load_profile_alphabet(base_weights)
        alphabet = build_alphabet(request.form.get("extra_chars", ""), base_alphabet, samples)
        def work(job_id, update):
            logger.info('Training job=%s profile=%s base=%s requested_device=%s '
                        'samples=%d epochs=%d lr=%g alphabet=%d',
                        job_id, profile_name, base_profile, device, len(samples), epochs, lr, len(alphabet))
            from ai.training.finetune import create_training_model, fit
            from ai.training.personalization import fit_personalized
            model = create_training_model(alphabet=alphabet, weights_path=base_weights, device=device)
            update(device=model.device, backend=getattr(model, 'backend', 'tensorflow'),
                   device_name=getattr(model, 'device_name', model.device),
                   base_profile=base_profile, learning_rate=lr, alphabet_size=len(alphabet))
            started = time.monotonic()
            def progress(metrics):
                update(status="training", **metrics)
            history, selection = fit_personalized(model, samples, epochs=epochs, lr=lr,
                                                 sample_progress_cb=progress, fit_fn=fit)
            output_path = os.path.join(PROFILES_DIR, f"{profile_name}.npz")
            logger.debug('Saving training profile job=%s path=%s', job_id, output_path)
            update(status="validating", progress=97)
            metadata = dict(selection, source_profile=base_profile, learning_rate=lr,
                            device=model.device, backend=getattr(model, 'backend', 'tensorflow'),
                            num_samples=len(samples), created_at=datetime.now(timezone.utc).isoformat())
            validation = save_validated_profile(model, output_path, samples=samples, metadata=metadata)
            logger.info('Training profile saved job=%s profile=%s path=%s', job_id, profile_name, output_path)
            _invalidate_model(profile_name)
            return {"profile_name": profile_name, "num_samples": len(samples), "epochs": len(history), "device": model.device,
                    "train_seconds": round(time.monotonic() - started, 1), "loss_start": round(history[0], 3),
                    "loss_end": round(history[-1], 3), "alphabet_size": len(alphabet),
                    "new_chars": [c for c in alphabet if c not in base_alphabet], "profiles": list_profiles(),
                    "generation_checks_passed": validation['passed'],
                    "style_samples_retained": validation.get('style_samples_retained', 0),
                    "style_warning": validation.get('style_warning'), **selection}
        job_id = _start_job("training", work)
        return jsonify(job_id=job_id, progress_url=f"/jobs/{job_id}"), 202
    except (ValueError, TypeError, KeyError) as error:
        return jsonify(error=str(error)), 400


@app.route("/capture-tool")
def capture_tool():
    return send_file(os.path.join(WEBAPP_DIR, "capture_tool", "index.html"))


@app.route("/calibration-pattern/<profile>.gcode")
def calibration_pattern(profile):
    printer = PrinterProfile()  # uses defaults; the pattern only needs pen/Z/speed fields
    gc = calibration_pattern_gcode(printer)
    path = os.path.join(DOWNLOADS_DIR, "calibration_pattern.gcode")
    with open(path, "w") as f:
        f.write(gc)
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
