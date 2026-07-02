# alphaXiv x marimo submission slice

This folder contains the competition-focused notebook slice for:

```text
Allowed Space Notebook:
Interactive Bootstrap Map for String Amplitude Rigidity
```

Selected alphaXiv paper:

```text
Bootstrap Principle for the Spectrum and Scattering of Strings
Clifford Cheung, Aaron Hillman, Grant N. Remmen
https://www.alphaxiv.org/abs/2406.02665
```

## Submission role

The official submission should be a molab/marimo notebook link plus a video
explainer under 5 minutes. The local React/Vite app is supporting material for
the video, not the submission URL.

This notebook turns the paper into an interactive assumption-risk map:

- background axioms: unitarity, crossing, analyticity
- stronger premises: Regge control, level truncation, ultrasoftness
- simplified `q/r/w` deformation controls
- allowed-space compression visualization
- residue-zero scaffold visualization
- research-platform handoff: claim, evidence note, risk critique, report note

## Guardrails

- Do not claim this proves string theory is the universe.
- Use "conditional rigidity" or "conditional uniqueness" language.
- Keep strong premises visible.
- Treat Assumption Risk as a first-class output, not a weakness to hide.
- Do not connect external APIs, real AI providers, or secrets.

## Run locally

Use Python 3.10 or newer.

```powershell
python -m venv $env:TEMP\s-matrix-qg-marimo-venv
& "$env:TEMP\s-matrix-qg-marimo-venv\Scripts\python.exe" -m pip install --upgrade pip
& "$env:TEMP\s-matrix-qg-marimo-venv\Scripts\python.exe" -m pip install -r notebooks\alphaxiv-marimo\requirements.txt
& "$env:TEMP\s-matrix-qg-marimo-venv\Scripts\python.exe" -m marimo check notebooks\alphaxiv-marimo\allowed_space_bootstrap.py
& "$env:TEMP\s-matrix-qg-marimo-venv\Scripts\python.exe" -m marimo edit notebooks\alphaxiv-marimo\allowed_space_bootstrap.py
```

For a read-only local app:

```powershell
& "$env:TEMP\s-matrix-qg-marimo-venv\Scripts\python.exe" -m marimo run notebooks\alphaxiv-marimo\allowed_space_bootstrap.py
```

## molab handoff

Final submission still needs to be uploaded/shared through molab:

1. Open molab.
2. Create or import the notebook from `allowed_space_bootstrap.py`.
3. Confirm the notebook runs from top to bottom.
4. Copy the molab notebook link.
5. Submit the molab link, the exact paper title, and the video explainer through
   the competition form.

## Optional GPU path

The parameter scan cell is GPU-ready when a CUDA PyTorch runtime is available.
It falls back to NumPy CPU execution when CUDA is unavailable. This keeps local
validation simple while allowing molab GPU review to exercise a real batched
parameter scan.
