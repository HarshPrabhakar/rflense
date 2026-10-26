# Development and validation

## Install

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[test,quality]"
python -m pytest
ruff check src tests
mypy src/rflens
```

Install the web UI with `python -m pip install -e ".[ui]"` and run `streamlit run src/rflens/app.py`. YAML validation config support is in the `validation` extra.

## Large captures

`welch_psd` streams overlapping FFT windows in batches. Complex float cfile, SigMF, and NPY samples use memory maps; integer formats are normalized to complex64 during load. Benchmark the PSD path on the current machine with:

```powershell
$env:PYTHONPATH = "src"
python examples/benchmark_large_capture.py --samples 10000000
```

Runtime varies by CPU, BLAS/FFT build, and memory speed. The script reports the observed result and does not encode a performance claim.

## Phase checkpoints

Each phase is committed independently. CI runs lint, type checking, and tests on Python 3.11 and 3.12. The synthetic validation snapshot and its low-SNR limitations are recorded in the README. A real-recording example requires an actual IQ capture; no real capture was present in the initial workspace.
