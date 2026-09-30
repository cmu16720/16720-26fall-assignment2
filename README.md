# 16-720 Computer Vision — Assignment 2

**Representations and Visual Recognition**

| Item | Value |
|---|---|
| Released | September 29, 2026 |
| Due | October 27, 2026 |
| Points | 100, plus up to 10 total extra-credit points |
| Expected effort | 12+ hours; report unexpected infrastructure overhead |
| Compute | Google Colab; GPU for standard training, CPU for smoke checks |
| Final artifact | Recognition Atlas and static Hero preview |

Implement a compact residual CNN, compare image representations, and investigate
recognition failures. Then build a Recognition Atlas using a frozen pretrained
model and three related categories of your choice.

## Getting started

Read the [assignment instructions (PDF)](assignment_instructions.pdf) first.
It is the authoritative source for requirements, grading, and policy.

- Work through the [starter notebook](notebooks/pa2_starter.ipynb).
- Complete the 12 `TODO(student)` regions in [`student/pa2.py`](student/pa2.py).
- Write responses in [`answer_template.tex`](answer_template.tex).
- Use [`tests/test_pa2_public.py`](tests/test_pa2_public.py) for public checks.
- See [`data/PROVENANCE.md`](data/PROVENANCE.md) for data and model sources.

## Colab workflow

1. Upload `notebooks/pa2_starter.ipynb` to Colab. Keep `RUN_MODE = "smoke"`
   while implementing; a CPU runtime is sufficient.
2. Run setup and upload the complete course-supplied `pa2_student.zip` when
   prompted. Setup extracts the release and installs its dependencies. If asked
   to restart, rerun the mode and setup cells; they reuse the extracted files.
3. Answer B1, then edit `student/pa2.py` in Colab's Files pane and save it.
   Run the notebook's reload and public-test cells after editing. The untouched
   starter is expected to fail until its TODOs are completed.
4. Back up your code before switching to a fresh GPU runtime. Select
   `RUN_MODE = "standard"`, run setup, and restore your edited `student/pa2.py`.
   Complete the 25-epoch run and the separate five-epoch C3 comparisons. Smoke
   results do not count as assignment evidence. Contact staff if a GPU is unavailable.
5. Complete Parts C–D, following the notebook's image-upload and split instructions.

Only marked regions of `student/pa2.py` should change. Add your responses,
experiments, and Atlas choices in the notebook, and complete the manifest and
written template as needed; keep the required training configuration unchanged.

Download edited code regularly: saving the notebook does not back up separate
runtime files. The final backup cell downloads code, manifests, and artifacts.
Download the executed notebook separately with **File → Download → Download .ipynb**.
Keep your source Atlas images backed up locally.

## Local checks (optional)

Use Python 3.10 or newer, from the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-colab.txt
.venv/bin/python -m pytest tests/test_pa2_public.py -q
```

If `uv` is unavailable, use a standard virtual environment and
`python -m pip install -r requirements-colab.txt`. Required standard training
still uses the Colab GPU workflow above.

## Written evidence and submission

Figures are saved under `artifacts/pa2_standard/`. Save added figures there too.
Upload the figures alongside `answer_template.tex` in Overleaf and replace each
relevant evidence box with `\includegraphics[width=\linewidth]{your_figure.png}`.
Compile with pdfLaTeX. Every graded figure/table must appear in both the executed
notebook and its labeled PDF section.

Submit the following, replacing `<andrewid>` with your Andrew ID:

- `pa2_<andrewid>.ipynb` — executed notebook;
- `pa2_<andrewid>.py` — a renamed copy of `student/pa2.py`;
- `pa2_written_<andrewid>.pdf`;
- `tinyresnet_<andrewid>.pt` — a copy of `artifacts/pa2_standard/tinyresnet_standard.pt`;
- `recognition_atlas_<andrewid>.html` or `.mp4`;
- `hero_<andrewid>.png`; and
- `atlas_manifest_<andrewid>.csv`.

Rename the generic Atlas/Hero exports before submission. The Atlas bundle is
limited to 25 MB, the optional narrated video to 60 seconds, and the Hero to
10 MB. Check the full submission specification in the PDF, including saved
plots, Gradescope page mapping, acknowledgments, and AI disclosure. Do not
submit raw private photographs, credentials, or pretrained-model caches.

## Collaboration and AI policy

Discuss ideas and debugging strategies, but produce your own submissions and
acknowledge collaborators. Generative AI is prohibited for **all of Parts A and B**.
It is permitted with disclosure in C–E, but may not replace or modify the required
Part B implementation. See the PDF for the complete policy.

For a capture alternative, contact staff before relying on it. The bundled
`data/accessibility_fallback/` requires a completed staff approval record and
waives only the approved photo-collection requirement, not the technical work.
