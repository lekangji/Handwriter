# AI improvements and profile management

Requested scope: automatic long-word recovery, best personalization checkpoint, captured-sample priming, CUDA inference, profile alphabet/details, default pinned first and persistent model ordering.

Preserve existing NPZ profiles, exports, calibration and user data. Leave the pre-existing uv.lock modification untouched. Implement in this workspace; do not merge or publish.

1. Layout: bounded token fragments with existing hyphens preferred, no added spaces within words, adaptive subdivision of rejected fragments only. Verify the reported word, long tokens, joins and bounded errors.
2. Training: optional epoch hooks in both backends, shared selection using complete transcription groups held out when at least five distinct phrases permit character coverage. Evaluate initial and improving checkpoints, require generation checks, restore best eligible checkpoint, stop after three unimproved epochs. Label tiny-dataset training evaluation honestly. Test selection, duplicates, rollback, early stopping and both backends.
3. Style: retain up to three short calibrated captures as model-space offsets in `.style.json`; validate compatibility, prime generation, account for priming-prefix alignment. Publish `.training.json` metadata and style together with NPZ/alphabet; preserve sidecars on rename/delete. Test persistence and alignment.
4. Inference: optional PyTorch CUDA device-resident sampler using existing weights and safeguards; retain NumPy CPU. Auto prefers available CUDA; explicit unavailable CUDA errors clearly. Cache by profile/device, add selector and resolved-device display. Test CPU forward parity and deterministic sampler; run CUDA smoke/benchmark when hardware permits.
5. Profiles: details expose character-level alphabet, missing capitals, training/checkpoint information and saved style status. Persist ordering in `.order.json`, keep default first and move other profiles up/down. Test unsafe inputs, persistence, rename/delete and default pinning.
6. Verification: complete Python and browser-script suites, actual generation probes, fresh review and README updates. Report hardware/environment limitations without asserting unmeasured speedups.

Review focus: duplicate validation leakage; rejected/final epochs replacing best weights; intra-word spaces; priming repeating the prefix; lost sidecars; missing CUDA; default displacement; stale order after renaming.

Completed verification: 86 Python tests ran, 85 passed and one TensorFlow GPU check skipped; all three JavaScript suites passed. Actual CUDA inference and end-to-end personalization passed on an RTX 3060. Warmed generation benchmark measured about 2.2x faster than NumPy CPU, with additional first-run CUDA preparation time. Review found that valid capture coordinates could still cause primed generation failures; saved primers now pass generation checks, with unprimed fallback if none qualify. `git diff --check` passed.
