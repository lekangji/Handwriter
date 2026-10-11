# AI Handwriter

Write typed documents in AI handwriting, preview the exact plotter strokes, and export SVG, PNG, or G-code.

## Run

```powershell
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The Flask entry point stays at the project root; web pages and assets live in `webapp/`.

## Structure

- `app.py`: local Flask server and background jobs.
- `webapp/templates/`, `webapp/static/`: web UI templates and shared assets.
- `ai/inference/hw_model.py`, `ai/inference/hw_weights.npz`: pretrained NumPy synthesis.
- `ai/training/samples.py`, `finetune.py`, `finetune_cli.py`: lightweight sample validation and TensorFlow adaptation.
- `document/`: page layout, calibration, previews, and G-code validation.
- `webapp/capture_tool/index.html`: standalone capture UI; also served at `/capture-tool`.
- `profiles/`: existing and newly trained weights with alphabet sidecars.
- `samples/`: preserved captured samples moved from the former webapp directory.
- `tests/`: generation, web jobs, personalization, and capture regression checks.

## Write and export

Paste text, choose a profile, and select **Create handwriting**. Generation starts in a background worker; the page stays usable and reports line progress and an estimated remaining time. Reloading reconnects to a running job while the server remains running.

**Generation device** accepts `auto`, `cpu`, `cuda`, or `cuda:N`. Auto prefers available PyTorch CUDA; CPU retains NumPy inference. Explicit CUDA requests report an error when the requested GPU is unavailable. CUDA uses the same NPZ profiles, keeps recurrent state and weights on the GPU, and replays cached CUDA graphs in blocks of 32 points. New sequence lengths/settings take extra time to prepare; subsequent draws reuse those graphs. The result shows the actual device used. Seeds are repeatable within a backend; CPU and CUDA can produce different stochastic draws.

Neatness controls output variance. **Fine-tune the feel** exposes chunk length, retry count, steps per character, letter width, gaps between chunks, and slant. Page setup controls physical letter size, ruling, margins, and paragraph spacing. Short chunks reduce attention drift; more retries and steps cost time. Physical line length follows usable paper width and letter width, without a fixed character cap. Multi-page documents join independent short chunks within each physical line. The generator rejects incomplete or runaway draws rather than returning rejected strokes. Failed multi-word chunks fall back to independent words, without restarting completed pages. Sample checks do not import TensorFlow; training preparation runs in the worker.

Unsupported characters are replaced or removed using the profile alphabet. The result shows a warning and the exact normalized text. Default weights lack uppercase Q, X, and Z; those fall back to lowercase. Inspect every preview: attention and geometry checks do not perform handwriting recognition and cannot guarantee exact transcription.

Words longer than the configured chunk length are split automatically, preferably at existing hyphens. Fragments preserve every supported character and use a small intra-word gap rather than adding a word space. If a draw fails, only that chunk is retried as words or smaller fragments; a fragment that still fails down to a single character reports a generation error. This recovery does not change your text or export rejected strokes.

Printer and paper coordinates still need calibration. SVG, PNG, and G-code use the same stroke geometry. Unsafe paper or printer coordinates block G-code export; previews remain available. A4 paper is taller than a default 220 mm printer bed, so full-page G-code may require a larger bed or smaller paper. The printer pauses between pages for manual paper changes.

## Personalize

1. Open **Capture samples**, calibrate a tall letter, and write the shown phrases with a stylus if possible. Use Undo to repair strokes. A browser draft saves completed samples; download JSON to keep them outside the browser.
2. Open **Personalize**, upload that JSON, choose a new profile name and source profile, and start training. Sample characters are added to the alphabet automatically.
3. Select the new profile on **Write & Export**. Existing profiles and default weights are preserved.

Capture converts browser y-down positions into the model's y-up offsets, preserves pen lifts and dots, and resamples by distance in calibrated model units. Merged sessions retain their physical scale. The training graph uses a dynamic TensorFlow loop, validates finite losses and gradients, and defaults to a conservative learning rate of `0.00001`. More epochs on very few samples can overfit; add varied clear phrases rather than simply increasing epochs.

Before a new profile is saved, its exported weights must complete four deterministic generation checks using the normal inference limits. Failed profiles are discarded with an error even if training NLL improved. These are basic generation checks, not a test of readability or personal style; inspect generated phrases you did not capture. Existing profiles are preserved.

Personalization now evaluates the source model and each improving epoch and restores the lowest-evaluation-NLL checkpoint that passes those generation checks. When at least five distinct phrases permit it, roughly 20% of transcription groups are held out; repeated copies of a phrase stay together and every captured character retains a training example. Smaller datasets, or datasets without a suitable holdout, use evaluation on training samples and label it explicitly. Training stops after three epochs without an eligible improvement. **Maximum epochs** is an upper limit; the result reports completed epochs, selected checkpoint (epoch 0 means the source model), evaluation kind and best NLL. Neither NLL nor these checks measures readability or style fidelity.

New profiles retain up to three compatible short captures as model-space offsets in `<name>.style.json`; generation primes its recurrent state with the first saved sample and emits only target strokes. `<name>.training.json` records checkpoint selection and training details. Older profiles remain usable without either sidecar; retrain under a new name to retain captures for priming. Keep all sidecars together when copying profiles between machines. They contain personal handwriting samples and should be treated as part of your profile data.

On **Profiles**, expand **Model details** to inspect the supported alphabet, missing capitals, retained style samples and recorded training details. This is a character model; it has no fixed word vocabulary. Default stays first in all lists. The arrow buttons persist the order of personal profiles in `profiles/.order.json`. Rename and delete also handle style/training sidecars and preserve the remaining order.

Retained style samples must also pass primed generation checks before they are enabled. If none passes, the working weights are saved without style priming and the training result explains why. This protects an otherwise usable profile from an incompatible capture; the checks still do not verify handwriting recognition.

Web training permits up to 800 sample-epochs. For larger datasets:

```powershell
python -m ai.training.finetune_cli handwriting_samples.json --name my_handwriting --epochs 20 --lr 0.00001
```

A grown source profile keeps its alphabet order; new characters are appended once. Calibrated exports preserve their recorded point sampling, including older files without a resampling marker: densifying them can disrupt pretrained text attention while making NLL look better. Uncalibrated files use a shared scale estimated from median phrase height and half-unit resampling. Profiles trained with the earlier densification, coordinate or alphabet bugs should be rebuilt from default under a new name. Fixing the pipeline does not repair those saved weights.

### CUDA and external GPUs

Training defaults to `auto`: PyTorch CUDA when available (including native Windows), otherwise TensorFlow GPU/CPU. Set **Training device** on Personalize or pass `--device cpu`, `--device cuda`, or `--device cuda:1` to the CLI. Explicit CUDA requests fail with setup instructions when the requested GPU is unavailable. The selected device appears in progress and results. Both training backends export the same NPZ profiles for NumPy inference.

PyTorch computes the output head and density loss for all pen points in a sample together, reducing repeated small operations. Recurrent state, sample order, float32 precision, per-sample Adam updates, and validation remain the same. Losses and weights can differ by normal float32 rounding.

For NVIDIA CUDA training on Windows, Linux, or WSL2, install CUDA-enabled PyTorch in the same environment used to run the app. A compatible NVIDIA driver is required; see the [PyTorch installation guide](https://pytorch.org/get-started/locally/). From your activated environment:

```bash
pip install -r requirements.txt
pip install -r requirements-cuda.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m ai.training.finetune_cli handwriting_samples.json --name my_cuda_profile --device cuda --epochs 20
# Or run the web app in the same environment:
python app.py
```

Restart the app after installation or code changes. A job already running on CPU stays on CPU; new jobs use the updated backend. Choose `cuda` to require GPU training instead of allowing CPU fallback. CUDA-enabled PyTorch is a normal project dependency. With uv, install and run using `uv sync` and `uv run python app.py`; no extra flag is needed.

An attached NVIDIA external GPU works when the OS and PyTorch expose it. Indices follow visible GPUs. Use `--device cuda:1` for the second GPU, or `CUDA_VISIBLE_DEVICES=1 python -m ai.training.finetune_cli handwriting_samples.json --name external_profile --device cuda` to expose only that GPU (then indexed as `cuda:0`). Training uses one selected GPU per job.

For a remote/cloud GPU, copy this project, the samples JSON, and the source profile's `.npz` and `.alphabet.json` to the GPU machine, install `requirements-cuda.txt`, and run the same CLI there. Copy both resulting profile files from `profiles/` back to your local `profiles/` folder. This is a CLI workflow; the local web app does not submit remote jobs.

### Backend logs

Successful job polling and static-file requests are quiet by default. Failed requests and server warnings remain visible. Training logs include the backend/device, dataset size, epochs, learning rate, epoch NLL and change, best NLL, timing, ETA, and throughput. Long epochs report sample progress at most once every 30 seconds. Profile saves and job failures include their job ID; failures retain their traceback. Sample text is not logged.

Set `HANDWRITER_LOG_LEVEL` to `DEBUG` for full request logs and extra job details, or `WARNING` to silence training summaries. The default is `INFO`. This applies to both the web app and training CLI. TensorFlow native startup information is hidden; warnings and errors remain visible. An explicit `TF_CPP_MIN_LOG_LEVEL` value takes precedence.

Training uses a tqdm sample progress bar in interactive terminals, with epoch and running NLL. Noninteractive output keeps bounded logging; environments without tqdm also retain that fallback. The training page receives live sample counts, motion-step counts, running NLL, best completed epoch NLL, elapsed time, estimated remaining time, processing rate, device/backend, source profile, learning rate and completed epoch history. Updates are throttled to twice per second, with epoch boundaries always reported; the page continues using its existing job polling.

## Checks

```powershell
python -m unittest discover -s tests -v
node tests/test_capture.cjs
node tests/test_jobs.cjs
```

Compare the previous and optimized PyTorch training steps without saving profiles:

```powershell
python tests/benchmark_training.py --device cuda
python tests/benchmark_training.py --device cpu
```

The benchmark uses synthetic 128- and 512-point samples, five warmup steps and the median of five full training steps (forward, backward, gradient clipping and Adam). It reports first-step time separately because TorchScript startup can affect short runs. Results depend on hardware and sample lengths.

Compare CPU and CUDA document generation, including automatic long-word splitting:

```powershell
python tests/benchmark_generation.py --device cuda --repeats 3
```

This benchmark excludes weight loading, rendering and export, reports cold and warmed timings separately, and does not save profiles. On the development RTX 3060, the three-line fixture took a warmed median of 0.217 seconds on CUDA versus 0.471 seconds on NumPy CPU (about 2.17× faster). The first CUDA draw took 2.480 seconds versus 0.488 seconds on CPU because of graph preparation. These are one local measurement with one CPU thread, not a guarantee for other documents or hardware; the backends also generated different point counts (1244 versus 1278).

TensorFlow tests skip when TensorFlow is unavailable. Flask and rendering dependencies are needed for web tests. No packages are downloaded by the tests.

The worker and job registry are for a local single-process app. Only one generation or training job runs at a time; server restarts discard job status. Runtime previews are retained for two hours, with at most 50 output jobs. `.backups` is not used or modified.
