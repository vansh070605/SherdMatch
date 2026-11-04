"""Gradio demo: AI-based archaeological pottery fragment matching.

Run:  python app.py      (after scripts/run_experiments.py has built the cache)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".pylibs"))  # gradio installed inside the project folder

from pfm.matcher import Matcher  # noqa: E402  (first: loads torch before cv2)
import gradio as gr  # noqa: E402

M = Matcher()
CHOICES = [n.replace("_exterior.png", "") for n in M.names]
METHODS = {"Proposed: appearance + fracture-edge geometry": "proposed",
           "Baseline: image-only similarity (DINOv2)": "baseline"}


def _results(query_frag, df, known_partners=None, k=5):
    gallery, rows = [], []
    for rank, (_, r) in enumerate(df.head(k).iterrows(), 1):
        g = int(r.other)
        hit = known_partners is not None and g in known_partners
        cap = f"#{rank} {CHOICES[g]}  score {r.score:.3f}" + ("  ✓ expert-confirmed refit" if hit else "")
        gallery.append((M.thumbnail(M.frags[g]), cap))
        rows.append([rank, CHOICES[g], round(float(r.score), 3), round(float(r.dino_cos), 3),
                     round(float(r.colour_sim), 3), int(r.contact), round(float(r.overlap), 3),
                     "yes" if hit else ("-" if known_partners is None else "no")])
    best = df.iloc[0]
    R, t = best.fit
    fit_img = M.render_fit(query_frag, M.frags[int(best.other)], R, t)
    return gallery, fit_img, rows


def run_gallery(choice, method):
    q = CHOICES.index(choice)
    df = M.rank_gallery_query(q, METHODS[method])
    partners = M.partners[q]
    gallery, fit_img, rows = _results(M.frags[q], df, partners)
    ranks = [i + 1 for i, o in enumerate(df.other) if o in partners]
    note = (f"Expert-confirmed partners of {choice}: {', '.join(CHOICES[p] for p in sorted(partners))}. "
            f"First true partner ranked #{ranks[0]} of {len(df)}.")
    return M.thumbnail(M.frags[q]), gallery, fit_img, rows, note


def run_upload(path, method):
    if path is None:
        raise gr.Error("Upload a fragment image first (PNG with transparent background, or a photo on a plain backdrop).")
    fq, df = M.rank_new_image(path, METHODS[method])
    gallery, fit_img, rows = _results(fq, df)
    return M.thumbnail(fq), gallery, fit_img, rows, "Uploaded fragment compared against the 124 gold-standard sherds."


HEAD = ["rank", "sherd", "match score", "DINOv2 cos", "colour sim", "edge contact (px)", "overlap", "true refit?"]

with gr.Blocks(title="Pottery Fragment Matching") as demo:
    gr.Markdown(
        "# AI-Based Archaeological Pottery Fragment Matching\n"
        "Pick a sherd (or upload one). The system ranks the most likely joining fragments from the "
        "Daxinzhuang gold-standard collection and previews the best physical fit. "
        "It *recommends* candidates for an expert to check; it does not confirm joins.\n\n"
        "<small>The demo model is trained on all 111 expert pairs, so scores for gallery sherds are optimistic. "
        "The report's numbers come from 5-fold vessel-grouped cross-validation.</small>")
    method = gr.Radio(list(METHODS), value=list(METHODS)[0], label="Matching method")
    with gr.Tabs():
        with gr.Tab("Gallery sherd"):
            with gr.Row():
                choice = gr.Dropdown(CHOICES, value="JDX-103", label="Query sherd")
                go1 = gr.Button("Find matches", variant="primary")
        with gr.Tab("Upload a fragment"):
            with gr.Row():
                up = gr.Image(type="filepath", label="Fragment image")
                go2 = gr.Button("Find matches", variant="primary")
    note = gr.Markdown()
    with gr.Row():
        qimg = gr.Image(label="Query", height=260)
        fit = gr.Image(label="Best fit: digital reconstruction preview (query = dark outline)", height=260)
    gal = gr.Gallery(label="Top-5 candidates", columns=5, height=280)
    table = gr.Dataframe(headers=HEAD, label="Why these candidates (feature evidence)")
    outs = [qimg, gal, fit, table, note]
    go1.click(run_gallery, [choice, method], outs)
    go2.click(run_upload, [up, method], outs)
    demo.load(run_gallery, [choice, method], outs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
