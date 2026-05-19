# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python research implementation of CoTRouter and related DE-Cascade experiments. Core routing logic lives in `cotrouter.py`; experiment orchestration is in `cotrouter_main.py`, `cotrouter_runner.py`, `runner.py`, and `cotrouter_param_sweep.py`. Shared configuration, model wrappers, data loading, and metrics are in `config.py`, `models.py`, `data.py`, and `utils.py`. Plotting and reporting scripts include `plot_results.py`, `www2plot.py`, `generate_latex_tables.py`, and `analyze_switch_points.py`. Generated artifacts are stored under `results/`, `case_entropy/`, and `case_entropy_tokens/`; avoid editing these unless updating documented experiment outputs.

## Build, Test, and Development Commands

No build step is required. Run scripts from the repository root.

```bash
python cotrouter_main.py --dataset GSM8K --experiment baselines --num_samples 20
```

Runs a small baseline experiment and writes timestamped output under `results/`.

```bash
bash run_test_gsm8k.sh
```

Runs the local smoke-test wrapper; update model paths and GPU IDs inside the script before use.

```bash
python cotrouter_param_sweep.py --dataset GSM8K --num_samples 100
python plot_results.py --results_dir results/cotrouter_YYYYMMDD_HHMMSS
```

Runs a parameter sweep, then generates figures from an existing result directory.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and standard `snake_case` names for functions, variables, and files. Keep dataclass-based configuration patterns consistent with `config.py`. Prefer explicit arguments over hidden globals for new experiment settings, but preserve existing environment-variable controls such as `GLOBAL_THRESHOLD` and `TARGET_LLM_TOKEN_RATIO`. Keep comments focused on non-obvious routing, evaluation, or GPU-resource behavior.

## Testing Guidelines

There is no dedicated unit-test framework in this checkout. Validate changes with a small `--num_samples` run before launching full experiments. For routing or metric changes, compare `results.json`, `results_summary.csv`, and printed summaries against a previous run. Add focused tests or lightweight checks when introducing parsing, answer extraction, or math-equivalence behavior.

## Commit & Pull Request Guidelines

Git history is not available in this mounted checkout, so use concise imperative commit messages such as `Add GSM8K smoke test` or `Fix entropy threshold update`. Pull requests should describe the experiment or bug fixed, list commands run, note model paths or GPU assumptions, and include before/after metrics or artifact paths when behavior changes.

## Security & Configuration Tips

Do not commit private model weights, credentials, or large temporary outputs. Treat hard-coded absolute model paths in shell scripts as local defaults; document required replacements in PR notes. Use `--num_samples` for quick validation before consuming multiple GPUs.
