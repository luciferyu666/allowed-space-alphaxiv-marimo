# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.12",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Allowed Space Notebook

    ## Interactive bootstrap map for string amplitude rigidity

    **Selected alphaXiv paper:** Bootstrap Principle for the Spectrum and Scattering of Strings

    This competition notebook turns one alphaXiv paper into a small, runnable assumption-risk laboratory. It does not try to prove that string theory is the universe.

    It demonstrates conditional rigidity in a controlled bootstrap setting:

    > Under strong, visible bootstrap assumptions, a broad deformation space can compress toward the Veneziano amplitude.

    The notebook is a competition-scale slice of the larger `s-matrix-qg-lab` research platform: assumptions become controls, controls move the allowed region, and the result is handed off as claim/evidence/risk/report notes.
    """)
    return


@app.cell
def _():
    PAPER_TITLE = "Bootstrap Principle for the Spectrum and Scattering of Strings"
    PAPER_AUTHORS = "Clifford Cheung, Aaron Hillman, Grant N. Remmen"
    PAPER_ALPHAXIV_URL = "https://www.alphaxiv.org/abs/2406.02665"
    PAPER_ARXIV_URL = "https://arxiv.org/abs/2406.02665"
    PAPER_CORE_RESULT = (
        "The Veneziano amplitude is isolated by a bootstrap problem when "
        "faster-than-power-law high-energy falloff and level truncation are "
        "added to the usual amplitude constraints. If the assumptions are "
        "weakened, a larger three-parameter family remains."
    )
    return (
        PAPER_ALPHAXIV_URL,
        PAPER_ARXIV_URL,
        PAPER_AUTHORS,
        PAPER_CORE_RESULT,
        PAPER_TITLE,
    )


@app.cell
def _(
    PAPER_ALPHAXIV_URL,
    PAPER_ARXIV_URL,
    PAPER_AUTHORS,
    PAPER_CORE_RESULT,
    PAPER_TITLE,
    mo,
):
    mo.md(
        f"""
        ## Paper hook

        **Selected alphaXiv paper:** [{PAPER_TITLE}]({PAPER_ALPHAXIV_URL})

        **Authors:** {PAPER_AUTHORS}

        **arXiv:** [{PAPER_ARXIV_URL}]({PAPER_ARXIV_URL})

        **Notebook question:** Can the paper's bootstrap logic be made
        tangible with a few controls: background axioms, stronger Regge
        assumptions, level truncation, and a simplified `q/r/w` deformation
        family?

        **Core result, simplified for this notebook:** {PAPER_CORE_RESULT}
        """
    )
    return


@app.cell
def _(np):
    BACKGROUND_AXIOMS = ("unitarity", "crossing", "analyticity")
    STRONG_PREMISES = ("regge_control", "level_truncation", "ultrasoftness")

    def clamp(value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    def format_percent(value):
        return f"{100.0 * clamp(value):.0f}%"

    def compute_metrics(states, q_value, r_value, w_value):
        background_count = sum(1 for key in BACKGROUND_AXIOMS if states[key])
        strong_count = sum(1 for key in STRONG_PREMISES if states[key])
        parameter_distance = float(
            np.sqrt((q_value - 1.0) ** 2 + r_value**2 + w_value**2)
        )
        normalized_distance = clamp(parameter_distance / 0.80)
        premise_pressure = strong_count / len(STRONG_PREMISES)
        background_pressure = background_count / len(BACKGROUND_AXIOMS)

        rigidity_score = clamp(
            0.08
            + 0.18 * background_pressure
            + 0.20 * float(states["regge_control"])
            + 0.22 * float(states["level_truncation"])
            + 0.22 * float(states["ultrasoftness"])
            + 0.22 * (1.0 - normalized_distance)
        )
        assumption_risk = clamp(
            0.08
            + 0.12 * background_pressure
            + 0.18 * float(states["regge_control"])
            + 0.22 * float(states["level_truncation"])
            + 0.28 * float(states["ultrasoftness"])
            + 0.08 * (1.0 - normalized_distance)
        )
        allowed_volume = clamp(
            1.0
            - 0.18 * background_pressure
            - 0.18 * float(states["regge_control"])
            - 0.24 * float(states["level_truncation"])
            - 0.25 * float(states["ultrasoftness"])
            + 0.20 * normalized_distance
        )
        three_parameter_freedom = clamp(
            1.0 - 0.55 * premise_pressure - 0.30 * (1.0 - normalized_distance)
        )

        if rigidity_score >= 0.72 and assumption_risk >= 0.50:
            scenario = "Veneziano-like corner under strong premises"
        elif allowed_volume >= 0.58:
            scenario = "Broad allowed family remains"
        else:
            scenario = "Compressed but not yet isolated"

        return {
            "background_count": background_count,
            "strong_count": strong_count,
            "parameter_distance": parameter_distance,
            "normalized_distance": normalized_distance,
            "rigidity_score": rigidity_score,
            "assumption_risk": assumption_risk,
            "allowed_volume": allowed_volume,
            "three_parameter_freedom": three_parameter_freedom,
            "scenario": scenario,
        }

    def synthetic_residue(t_values, level, q_value, r_value, w_value):
        residue = np.ones_like(t_values, dtype=float)
        for pole_index in range(1, int(level) + 1):
            shifted_zero = q_value * pole_index + r_value
            residue *= (t_values + shifted_zero) / pole_index
        envelope = np.exp(-0.025 * abs(w_value) * t_values**2)
        return np.clip(residue * envelope, -8.0, 8.0)

    def run_parameter_scan(resolution, w_value, states):
        resolution = int(resolution)
        strong_bonus = (
            0.18 * float(states["regge_control"])
            + 0.22 * float(states["level_truncation"])
            + 0.22 * float(states["ultrasoftness"])
        )
        background_bonus = (
            sum(1 for key in BACKGROUND_AXIOMS if states[key])
            / len(BACKGROUND_AXIOMS)
            * 0.18
        )

        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            device = torch.device("cuda")
            q_values = torch.linspace(0.55, 1.45, resolution, device=device)
            r_values = torch.linspace(-0.45, 0.45, resolution, device=device)
            q_grid, r_grid = torch.meshgrid(q_values, r_values, indexing="xy")
            distance = torch.sqrt((q_grid - 1.0) ** 2 + r_grid**2 + w_value**2)
            normalized = torch.clamp(distance / 0.80, 0.0, 1.0)
            rigidity = torch.clamp(
                0.08 + background_bonus + strong_bonus + 0.22 * (1.0 - normalized),
                0.0,
                1.0,
            )
            risk = torch.clamp(
                0.12
                + 0.22 * float(states["level_truncation"])
                + 0.28 * float(states["ultrasoftness"])
                + 0.08 * (1.0 - normalized),
                0.0,
                1.0,
            )
            return {
                "q_values": q_values.detach().cpu().numpy(),
                "r_values": r_values.detach().cpu().numpy(),
                "rigidity": rigidity.detach().cpu().numpy(),
                "risk": risk.detach().cpu().numpy(),
                "backend": f"torch CUDA: {torch.cuda.get_device_name(0)}",
                "resolution": resolution,
            }
        except Exception:
            q_values = np.linspace(0.55, 1.45, resolution)
            r_values = np.linspace(-0.45, 0.45, resolution)
            q_grid, r_grid = np.meshgrid(q_values, r_values, indexing="xy")
            distance = np.sqrt((q_grid - 1.0) ** 2 + r_grid**2 + w_value**2)
            normalized = np.clip(distance / 0.80, 0.0, 1.0)
            rigidity = np.clip(
                0.08 + background_bonus + strong_bonus + 0.22 * (1.0 - normalized),
                0.0,
                1.0,
            )
            risk = np.clip(
                0.12
                + 0.22 * float(states["level_truncation"])
                + 0.28 * float(states["ultrasoftness"])
                + 0.08 * (1.0 - normalized),
                0.0,
                1.0,
            )
            return {
                "q_values": q_values,
                "r_values": r_values,
                "rigidity": rigidity,
                "risk": risk,
                "backend": "numpy CPU fallback",
                "resolution": resolution,
            }

    return (
        compute_metrics,
        format_percent,
        run_parameter_scan,
        synthetic_residue,
    )


@app.cell
def _(mo):
    unitarity_control = mo.ui.checkbox(value=True, label="Unitarity")
    crossing_control = mo.ui.checkbox(value=True, label="Crossing symmetry")
    analyticity_control = mo.ui.checkbox(value=True, label="Analyticity")
    regge_control = mo.ui.checkbox(value=True, label="Regge control")
    level_truncation_control = mo.ui.checkbox(
        value=True, label="Level truncation"
    )
    ultrasoftness_control = mo.ui.checkbox(value=False, label="Ultrasoftness")
    q_control = mo.ui.slider(
        start=0.55, stop=1.45, step=0.01, value=1.0, label="q deformation"
    )
    r_control = mo.ui.slider(
        start=-0.45, stop=0.45, step=0.01, value=0.0, label="r deformation"
    )
    w_control = mo.ui.slider(
        start=-0.45, stop=0.45, step=0.01, value=0.0, label="w deformation"
    )
    level_control = mo.ui.slider(
        start=3, stop=16, step=1, value=7, label="Residue level n"
    )
    grid_resolution_control = mo.ui.slider(
        start=35, stop=111, step=2, value=61, label="Parameter scan resolution"
    )

    mo.output.replace(
        mo.vstack(
            [
                mo.md("## Assumption controls"),
                mo.md(
                    "Toggle the background axioms and stronger premises. Move "
                    "`q/r/w` away from the Veneziano-like point `(1, 0, 0)` to "
                    "watch the allowed space reopen."
                ),
                mo.hstack(
                    [unitarity_control, crossing_control, analyticity_control],
                    justify="start",
                    gap=1,
                ),
                mo.hstack(
                    [
                        regge_control,
                        level_truncation_control,
                        ultrasoftness_control,
                    ],
                    justify="start",
                    gap=1,
                ),
                mo.hstack([q_control, r_control, w_control], justify="start", gap=1),
                mo.hstack(
                    [level_control, grid_resolution_control], justify="start", gap=1
                ),
            ]
        )
    )

    return (
        analyticity_control,
        crossing_control,
        grid_resolution_control,
        level_control,
        level_truncation_control,
        q_control,
        r_control,
        regge_control,
        ultrasoftness_control,
        unitarity_control,
        w_control,
    )


@app.cell
def _(
    analyticity_control,
    compute_metrics,
    crossing_control,
    level_truncation_control,
    q_control,
    r_control,
    regge_control,
    ultrasoftness_control,
    unitarity_control,
    w_control,
):
    assumption_states = {
        "unitarity": bool(unitarity_control.value),
        "crossing": bool(crossing_control.value),
        "analyticity": bool(analyticity_control.value),
        "regge_control": bool(regge_control.value),
        "level_truncation": bool(level_truncation_control.value),
        "ultrasoftness": bool(ultrasoftness_control.value),
    }
    metrics = compute_metrics(
        assumption_states,
        float(q_control.value),
        float(r_control.value),
        float(w_control.value),
    )
    return assumption_states, metrics


@app.cell
def _(format_percent, metrics, mo, q_control, r_control, w_control):
    mo.md(
        f"""
        ## Current scenario

        **Scenario:** {metrics["scenario"]}

        | Measure | Value | Interpretation |
        | --- | ---: | --- |
        | String rigidity | {format_percent(metrics["rigidity_score"])} | How strongly this toy model compresses toward the Veneziano-like point |
        | Assumption risk | {format_percent(metrics["assumption_risk"])} | How much the conclusion depends on stronger premises |
        | Allowed volume | {format_percent(metrics["allowed_volume"])} | How much deformation space remains open |
        | Three-parameter freedom | {format_percent(metrics["three_parameter_freedom"])} | How visible the broader `q/r/w` family remains |

        Current deformation point: `q={float(q_control.value):.2f}`,
        `r={float(r_control.value):.2f}`, `w={float(w_control.value):.2f}`.
        """
    )
    return


@app.cell
def _(metrics, plt):
    fig_allowed, ax_allowed = plt.subplots(figsize=(7.5, 4.8))
    prototypes = [
        ("Generic amplitude", 0.18, 0.18),
        ("Coon-like branch", 0.45, 0.32),
        ("Hypergeometric-like branch", 0.55, 0.40),
        ("Compressed bootstrap sector", 0.73, 0.62),
        ("Veneziano-like point", 0.91, 0.77),
    ]
    for label, rigidity, risk in prototypes:
        ax_allowed.scatter(
            risk,
            rigidity,
            s=90,
            color="#61738f" if "Veneziano" not in label else "#d15b41",
            edgecolor="#1f2937",
            linewidth=0.8,
        )
        ax_allowed.annotate(
            label,
            (risk, rigidity),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax_allowed.scatter(
        metrics["assumption_risk"],
        metrics["rigidity_score"],
        s=210,
        marker="*",
        color="#f0b429",
        edgecolor="#1f2937",
        linewidth=1.0,
        label="current controls",
    )
    ax_allowed.set_xlim(0, 1.0)
    ax_allowed.set_ylim(0, 1.0)
    ax_allowed.set_xlabel("Assumption risk")
    ax_allowed.set_ylabel("String rigidity")
    ax_allowed.set_title("Allowed space as assumptions strengthen")
    ax_allowed.grid(True, alpha=0.25)
    ax_allowed.legend(loc="lower right")
    fig_allowed.tight_layout()
    fig_allowed
    return


@app.cell
def _(
    assumption_states,
    grid_resolution_control,
    mo,
    run_parameter_scan,
    w_control,
):
    scan_result = run_parameter_scan(
        int(grid_resolution_control.value), float(w_control.value), assumption_states
    )
    mo.md(
        f"""
        ## Parameter scan

        The heatmap below scans a simplified `q/r` deformation plane at fixed
        `w={float(w_control.value):.2f}`. It is intentionally a toy model:
        the goal is to make the paper's qualitative logic inspectable, not to
        claim a full physics reproduction.

        **Backend:** `{scan_result["backend"]}` at
        `{scan_result["resolution"]} x {scan_result["resolution"]}` points.

        On a GPU molab runtime, this cell will use CUDA through PyTorch if it
        is available; otherwise it stays reproducible with NumPy.
        """
    )
    return (scan_result,)


@app.cell
def _(metrics, plt, q_control, r_control, scan_result):
    q_values = scan_result["q_values"]
    r_values = scan_result["r_values"]
    rigidity_grid = scan_result["rigidity"]
    risk_grid = scan_result["risk"]

    fig_scan, (ax_heat, ax_risk) = plt.subplots(1, 2, figsize=(11, 4.6))
    heatmap = ax_heat.imshow(
        rigidity_grid,
        origin="lower",
        extent=[q_values.min(), q_values.max(), r_values.min(), r_values.max()],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax_heat.scatter(
        [float(q_control.value)],
        [float(r_control.value)],
        marker="*",
        s=170,
        color="#f0b429",
        edgecolor="#111827",
        linewidth=0.8,
    )
    ax_heat.scatter([1.0], [0.0], marker="o", s=70, color="#f7f7f7", edgecolor="#111827")
    ax_heat.set_xlabel("q")
    ax_heat.set_ylabel("r")
    ax_heat.set_title("Rigidity score over deformation plane")
    fig_scan.colorbar(heatmap, ax=ax_heat, fraction=0.046, pad=0.04)

    risk_contours = ax_risk.contour(
        q_values,
        r_values,
        risk_grid,
        levels=[0.25, 0.40, 0.55, 0.70],
        colors=["#9ca3af", "#6b7280", "#374151", "#111827"],
        linewidths=1.1,
    )
    ax_risk.clabel(risk_contours, inline=True, fontsize=8)
    ax_risk.scatter(
        [float(q_control.value)],
        [float(r_control.value)],
        marker="*",
        s=170,
        color="#f0b429",
        edgecolor="#111827",
        linewidth=0.8,
    )
    ax_risk.scatter([1.0], [0.0], marker="o", s=70, color="#f7f7f7", edgecolor="#111827")
    ax_risk.set_xlabel("q")
    ax_risk.set_ylabel("r")
    ax_risk.set_title(
        f"Assumption-risk contours; allowed volume {metrics['allowed_volume']:.2f}"
    )
    fig_scan.tight_layout()
    fig_scan
    return


@app.cell
def _(
    level_control,
    np,
    plt,
    q_control,
    r_control,
    synthetic_residue,
    w_control,
):
    level = int(level_control.value)
    t_values = np.linspace(-level - 1.2, 1.5, 600)
    veneziano_residue = synthetic_residue(t_values, level, 1.0, 0.0, 0.0)
    deformed_residue = synthetic_residue(
        t_values,
        level,
        float(q_control.value),
        float(r_control.value),
        float(w_control.value),
    )

    fig_residue, ax_residue = plt.subplots(figsize=(9, 4.5))
    ax_residue.plot(
        t_values,
        veneziano_residue,
        color="#334155",
        linewidth=2.2,
        label="Veneziano-like residue zeros",
    )
    ax_residue.plot(
        t_values,
        deformed_residue,
        color="#d15b41",
        linewidth=2.0,
        linestyle="--",
        label="current q/r/w deformation",
    )
    for zero_index in range(1, level + 1):
        ax_residue.axvline(-zero_index, color="#cbd5e1", linewidth=0.8, alpha=0.8)
    ax_residue.axhline(0, color="#111827", linewidth=0.9)
    ax_residue.set_ylim(-8.0, 8.0)
    ax_residue.set_xlabel("momentum transfer proxy t")
    ax_residue.set_ylabel("normalized residue proxy")
    ax_residue.set_title(
        f"Mandated residue-zero scaffold at level n={level}"
    )
    ax_residue.legend(loc="upper right")
    ax_residue.grid(True, alpha=0.20)
    fig_residue.tight_layout()
    fig_residue
    return


@app.cell
def _(assumption_states, format_percent, metrics):
    missing_background = [
        key for key in ("unitarity", "crossing", "analyticity") if not assumption_states[key]
    ]
    strong_premises = [
        key.replace("_", " ")
        for key in ("regge_control", "level_truncation", "ultrasoftness")
        if assumption_states[key]
    ]
    support_status = (
        "high"
        if metrics["rigidity_score"] >= 0.72 and not missing_background
        else "conditional"
    )
    caution = (
        "The result is visually compelling but assumption-heavy."
        if metrics["assumption_risk"] >= 0.55
        else "The current setting keeps room for alternative deformations."
    )

    claim_draft = (
        "Within this simplified bootstrap map, the selected assumptions "
        f"produce {support_status} support for a Veneziano-like corner."
    )
    evidence_note = (
        "The heatmap and residue-zero plot show how stronger Regge behavior, "
        "level truncation, and closeness to q=1, r=0, w=0 compress the toy "
        "deformation family."
    )
    risk_critique = (
        f"Assumption Risk is {format_percent(metrics['assumption_risk'])}. "
        f"Visible strong premises: {', '.join(strong_premises) or 'none'}. "
        f"{caution}"
    )
    report_appendix = (
        "Appendix note: This notebook is an explanatory model for the paper's "
        "bootstrap logic. It is not a substitute for the paper's analytic "
        "proof and does not establish an unconditional physical uniqueness "
        "claim."
    )

    return claim_draft, evidence_note, report_appendix, risk_critique


@app.cell
def _(claim_draft, evidence_note, mo, report_appendix, risk_critique):
    mo.md(
        f"""
        ## Research-platform handoff

        The larger `s-matrix-qg-lab` platform tracks claims, evidence,
        assumption risk, and report appendices. This notebook produces the
        same kind of handoff in compact form.

        **Claim draft**

        {claim_draft}

        **Evidence note**

        {evidence_note}

        **Assumption-risk critique**

        {risk_critique}

        **Report appendix summary**

        {report_appendix}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Guardrails for reviewers

    - This notebook is centered on one alphaXiv paper, not on the full React/Vite research platform.
    - The computation is a reproducible explanatory model, not a complete reproduction of the paper.
    - "Rigidity" always means conditional rigidity inside the selected bootstrap sector.
    - Strong premises are displayed because they are part of the result, not background assumptions to hide.
    - The local web app at `http://127.0.0.1:5173/` is supporting demo material for the video, not the competition submission link.
    """)
    return


if __name__ == "__main__":
    app.run()
