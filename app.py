"""SherdMatch demo: a multi-page Gradio app.

Pages: Home (the problem) · How it works · Matcher · Results
Run:  python app.py      (after scripts/run_experiments.py has built the cache)
"""
import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".pylibs"))  # gradio installed inside the project folder

from pfm.matcher import Matcher  # noqa: E402  (first: loads torch before cv2)
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"
M = Matcher()
R = json.loads((OUT / "results.json").read_text())
DS = json.loads((OUT / "dataset_summary.json").read_text())
METHODS = {m["method"]: m for m in R["methods"]}
PROP, BASE, APPLR = ("Proposed: appearance + geometry (LR)", "DINOv2 embedding (cosine)", "Appearance features (LR)")
NAMES = [n.replace("_exterior.png", "") for n in M.names]
MODES = {"Proposed: appearance + edge geometry": "proposed", "Baseline: image-only (DINOv2)": "baseline"}


# ------------------------------------------------------------------ images
def data_uri(img, fmt="PNG", max_side=None):
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else Image.open(img)
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    if fmt == "JPEG":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, fmt, quality=85) if fmt == "JPEG" else im.save(buf, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def cutout(frag):
    """Sherd on a transparent background, cropped to its outline."""
    ys, xs = np.nonzero(frag.mask)
    rgba = np.dstack([frag.rgb, frag.mask.astype(np.uint8) * 255])
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


CUT = {}


def cut_uri(i, side=220):
    if (i, side) not in CUT:
        CUT[(i, side)] = data_uri(cutout(M.frags[i]), max_side=side)
    return CUT[(i, side)]


def fig(name, alt, max_side=1600):
    return f'<figure class="sm-fig"><img src="{data_uri(FIG / name, "JPEG", max_side)}" alt="{alt}"></figure>'


def transparent(rgb):
    """Make the white background of a rendered fit transparent."""
    alpha = (rgb.min(axis=2) < 248).astype(np.uint8) * 255
    return np.dstack([rgb, alpha])


# ------------------------------------------------------------------ copy
def pct(x):
    return f"{100 * x:.0f}%"


P, Bm = METHODS[PROP], METHODS[BASE]
d10 = R["bootstrap"]["proposed_vs_dinov2"]["hit@10"]
pairs_all = 17635 * 17634 // 2
idx = {n: i for i, n in enumerate(NAMES)}
# --r final rotation, --dx/--dy/--r0 scattered start of the fly-in, --d delay, --z parallax depth
HERO_SHERDS = [("JDX-100", "left:2%;top:4%;width:42%;--r:-6deg;--dx:-120px;--dy:-60px;--r0:-40deg;--d:.15s;--z:14"),
               ("JDX-103", "left:52%;top:0%;width:34%;--r:8deg;--dx:140px;--dy:-90px;--r0:50deg;--d:.3s;--z:22"),
               ("JDX-33", "left:6%;top:52%;width:36%;--r:4deg;--dx:-150px;--dy:110px;--r0:-30deg;--d:.45s;--z:10"),
               ("JDX-107", "left:46%;top:42%;width:40%;--r:-3deg;--dx:160px;--dy:120px;--r0:35deg;--d:.6s;--z:18"),
               ("JDX-118", "left:36%;top:28%;width:20%;--r:12deg;--dx:0px;--dy:-160px;--r0:90deg;--d:.8s;--z:30")]
hero_imgs = "".join(f'<img class="sm-float" style="{st}" src="{cut_uri(idx[n], 520)}" alt="pottery sherd {n}">'
                    for n, st in HERO_SHERDS)

HOME = f"""
<section class="sm-hero">
  <div class="sm-hero-text">
    <p class="sm-kicker">21CSE451T · Pattern Recognition case study</p>
    <h1>Sherd<span>Match</span></h1>
    <p class="sm-sub">AI-Based Archaeological Pottery Fragment Matching</p>
    <p class="sm-lead">Excavations recover pottery as hundreds of broken sherds. SherdMatch looks at one sherd and
    ranks every other sherd in the collection by how likely the two pieces are to <em>physically fit together</em>.
    That turns an all-against-all search into a short list for an archaeologist to check.</p>
    <div class="sm-cta"><a class="sm-btn" href="/match">Try the matcher →</a>
    <a class="sm-btn ghost" href="/method">How it works</a></div>
  </div>
  <div class="sm-hero-art">{hero_imgs}</div>
</section>

<section class="sm-section">
  <p class="sm-kicker">The problem</p>
  <h2>Thousands of fragments, one lost object</h2>
  <div class="sm-grid3">
    <div class="sm-card"><div class="sm-big sm-count">17,635</div><p>sherds catalogued from a single refuse pit at the
      Daxinzhuang site (Jinan, China).</p></div>
    <div class="sm-card"><div class="sm-big sm-count">{pairs_all / 1e6:.0f} M</div><p>possible pairs to compare by hand.
      The work grows with the <em>square</em> of the collection size.</p></div>
    <div class="sm-card"><div class="sm-big sm-count">1.5%</div><p>of sherd pairs actually join, even in a curated set.
      Almost every comparison is a dead end.</p></div>
  </div>
  <div class="sm-grid2">
    <div class="sm-note"><h3>Why it is hard</h3><ul>
      <li><b>Look-alike ware.</b> Most sherds are the same grey-to-orange cord-marked pottery.</li>
      <li><b>Eroded edges.</b> 3,000 years in the ground chip and wear the fracture edges.</li>
      <li><b>No fixed orientation.</b> Any sherd can be photographed at any rotation.</li></ul></div>
    <div class="sm-note"><h3>The question we test</h3><p>Does adding <b>fracture-edge geometry</b> to <b>image
      similarity</b> find joining sherds better than image similarity alone? Two pieces can look alike but not
      fit, or fit by coincidence.</p></div>
  </div>
</section>

<section class="sm-section">
  <p class="sm-kicker">The approach</p>
  <h2>Two kinds of evidence, one ranked list</h2>
  <ol class="sm-steps">
    <li><span>01</span><h4>Clean up</h4><p>Isolate the sherd, normalise lighting, trace its outline.</p></li>
    <li><span>02</span><h4>Appearance</h4><p>Colour, texture (LBP) and deep features (ResNet-50, DINOv2).</p></li>
    <li><span>03</span><h4>Edge fit</h4><p>Slide one outline along the other, align and check that the pieces touch
      without overlapping.</p></li>
    <li><span>04</span><h4>Rank</h4><p>A logistic-regression model turns 12 cues into P(join) and sorts the
      collection.</p></li>
  </ol>
</section>

<section class="sm-section">
  <p class="sm-kicker">What it achieves</p>
  <h2>Evaluated on {DS['gold_refit_pairs']} expert-confirmed joins</h2>
  <div class="sm-grid3">
    <div class="sm-card accent"><div class="sm-big sm-count">{pct(P['hit@10'])}</div><p>of sherds have a true partner among
      the top 10 suggestions (image-only: {pct(Bm['hit@10'])}).</p></div>
    <div class="sm-card"><div class="sm-big sm-count">+{100 * d10[0]:.1f}</div><p>points over the image-only plan at top 10,
      95% CI {100 * d10[1]:+.1f} to {100 * d10[2]:+.1f}.</p></div>
    <div class="sm-card"><div class="sm-big">{pct(R['sim']['pose_success_rate_50px'])} vs
      {pct(R['gold_pose']['pose_success_rate_50px'])}</div><p>edge-fit success on clean synthetic breaks vs real eroded
      sherds: the honest limit of geometry.</p></div>
  </div>
  <p class="sm-foot">Data: Daxinzhuang-18K (Ji et al., 2026, Zenodo 21553799, CC-BY 4.0) · Vansh Agrawal ·
  RA2311026010120 · Faculty: Dr Nithyashri J ·
  <a href="https://github.com/vansh070605/SherdMatch">github.com/vansh070605/SherdMatch</a></p>
</section>
"""

METHOD = f"""
<section class="sm-section first">
  <p class="sm-kicker">How it works</p>
  <h2>From a photograph to a ranked list of candidates</h2>
  <p class="sm-lead">Every sherd is described once and cached. When you ask for matches, SherdMatch compares the query
  with every sherd in the collection using two independent kinds of evidence, then a small, interpretable model
  weighs them.</p>
  {fig("method_architecture.png", "System architecture")}
</section>
<section class="sm-section sm-step">
  <div class="sm-stephead"><span>01</span><div><h3>Preprocessing</h3><p>The sherd is separated from the background
  (alpha mask, or Otsu + GrabCut for raw photos, rejecting the scale card). Lighting is evened out with CLAHE, and
  the outline is traced and resampled every 3 px. Scale is never changed per image, because edge lengths must
  match between pieces.</p></div></div>
  {fig("preprocessing_pipeline.png", "Preprocessing steps")}
</section>
<section class="sm-section sm-step">
  <div class="sm-stephead"><span>02</span><div><h3>Appearance cues</h3><p>Sherds from one vessel share their firing
  colour and surface texture. Four similarities are computed per pair: a Lab colour histogram, rotation-invariant LBP
  texture, and embeddings from ResNet-50 and DINOv2 averaged over four rotations (a sherd has no "up").</p></div></div>
  {fig("gold_samples.png", "Gold-standard sherds")}
</section>
<section class="sm-section sm-step">
  <div class="sm-stephead"><span>03</span><div><h3>Fracture-edge fit</h3><p>Where two sherds join, their outlines
  run in opposite directions and one piece's bump is the other's dent. SherdMatch compares curvature windows of
  sherd A with reversed, mirrored windows of sherd B. It aligns the best candidates with the Kabsch method, refines
  them with ICP, and keeps the fit with the most edge contact and no overlap.</p></div></div>
  {fig("method_edge_matcher.png", "How the edge matcher works")}
</section>
<section class="sm-section sm-step">
  <div class="sm-stephead"><span>04</span><div><h3>Deciding and ranking</h3><p>Twelve pair features (4 appearance,
  8 geometry) go into a class-balanced logistic regression: P(join) = σ(wᵀz + b). It was chosen over a random
  forest and an SVM because it ranked best, suits only 111 positive examples, and shows how much each cue
  counts.</p></div></div>
  {fig("results_lr_coefficients.png", "Model weights", 1100)}
</section>
<section class="sm-section sm-step">
  <div class="sm-stephead"><span>05</span><div><h3>Fair evaluation</h3><p>Sherds are grouped into the
  {DS['gold_vessel_groups']} vessels they come from, and whole vessels are held out in 5-fold cross-validation, so
  the model can never succeed by memorising a vessel's colour. Differences are tested with paired bootstrap
  confidence intervals.</p></div></div>
</section>
"""

rows = "".join(
    f"<tr class='{'hl' if n == PROP else ''}'><td>{lab}</td><td>{METHODS[n]['hit@1']:.3f}</td>"
    f"<td>{METHODS[n]['hit@5']:.3f}</td><td>{METHODS[n]['hit@10']:.3f}</td><td>{METHODS[n]['mrr']:.3f}</td></tr>"
    for n, lab in [("Random ranking", "Random ranking"), ("Colour histogram (chi2)", "Colour histogram"),
                   ("ResNet-50 embedding (cosine)", "ResNet-50"), (BASE, "DINOv2 (image-only baseline)"),
                   ("Geometry features (LR)", "Geometry only"), (APPLR, "Appearance (learned)"),
                   (PROP, "SherdMatch: appearance + geometry")])
b_app = R["bootstrap"]["proposed_vs_appearanceLR"]
RESULTS = f"""
<section class="sm-section first">
  <p class="sm-kicker">Results</p>
  <h2>What the experiments show</h2>
  <p class="sm-lead">124 query sherds, each ranked against the other 123, under vessel-grouped cross-validation.
  Hit@k = share of queries with a true partner among the top k suggestions.</p>
  <div class="sm-grid2 wide">
    <table class="sm-table"><thead><tr><th>Method</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th></tr>
    </thead><tbody>{rows}</tbody></table>
    {fig("results_cmc.png", "Cumulative match curve", 1000)}
  </div>
</section>
<section class="sm-section">
  <div class="sm-grid2">
    <div class="sm-note"><h3>Better than image-only</h3><p>SherdMatch puts a true partner in the top 10 for
    <b>{pct(P['hit@10'])}</b> of sherds against <b>{pct(Bm['hit@10'])}</b> for DINOv2 similarity alone. That is a
    significant gain of {100 * d10[0]:+.1f} points (95% CI {100 * d10[1]:+.1f} to {100 * d10[2]:+.1f}).</p></div>
    <div class="sm-note warn"><h3>…but geometry adds little</h3><p>Against learned appearance alone, adding edge
    geometry changes Hit@10 by {100 * b_app['hit@10'][0]:+.1f} points (not significant) and Hit@1 by
    {100 * b_app['hit@1'][0]:+.1f} (significant). The hypothesis is only partly supported.</p></div>
  </div>
</section>
<section class="sm-section">
  <h3 class="sm-h3">Why: eroded edges</h3>
  <p class="sm-lead">The same edge matcher recovers the true joint for
  {pct(R['sim']['pose_success_rate_50px'])} of clean synthetic breaks but only
  {pct(R['gold_pose']['pose_success_rate_50px'])} of real sherds. On worn edges a wrong fit often touches along more
  edge than the true one.</p>
  {fig("results_alignment_examples.png", "Alignment successes and failures")}
</section>
<section class="sm-section">
  <h3 class="sm-h3">Which cues matter</h3>
  {fig("results_ablation.png", "Ablation", 1200)}
</section>
<section class="sm-section">
  <h3 class="sm-h3">Examples</h3>
  {fig("results_qualitative_retrieval.png", "Qualitative retrieval", 1400)}
</section>
"""


# ------------------------------------------------------------------ matcher logic
def candidates_html(df, partners=None, k=5):
    cards = []
    top = float(df.score.iloc[0]) or 1.0
    for rank, (_, r) in enumerate(df.head(k).iterrows(), 1):
        g = int(r.other)
        hit = partners is not None and g in partners
        width = max(4, 100 * float(r.score) / top) if top > 0 else 0
        cards.append(
            f'<div class="sm-cand {"hit" if hit else ""}"><div class="sm-rank">#{rank}</div>'
            f'<img src="{cut_uri(g)}" alt="{NAMES[g]}"><div class="sm-cname">{NAMES[g]}</div>'
            f'<div class="sm-bar"><i style="width:{width:.0f}%"></i></div>'
            f'<div class="sm-score">score {float(r.score):.3f}</div>'
            + ('<div class="sm-badge">✓ expert-confirmed join</div>' if hit else "") + "</div>")
    return '<div class="sm-cands">' + "".join(cards) + "</div>"


def evidence_html(df, partners=None, k=5):
    body = "".join(
        f"<tr class='{'hl' if partners and int(r.other) in partners else ''}'><td>#{i}</td><td>{NAMES[int(r.other)]}</td>"
        f"<td>{float(r.score):.3f}</td><td>{float(r.dino_cos):.3f}</td><td>{float(r.colour_sim):.3f}</td>"
        f"<td>{int(r.contact)}</td><td>{float(r.overlap):.1%}</td>"
        f"<td>{'yes' if partners and int(r.other) in partners else ('–' if partners is None else 'no')}</td></tr>"
        for i, (_, r) in enumerate(df.head(k).iterrows(), 1))
    return ("<table class='sm-table small'><thead><tr><th>Rank</th><th>Sherd</th><th>Score</th><th>DINOv2 sim.</th>"
            "<th>Colour sim.</th><th>Edge contact (px)</th><th>Overlap</th><th>True join?</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def query_html(img_uri, title, lines):
    return (f'<div class="sm-query"><img src="{img_uri}" alt="query sherd"><div><p class="sm-kicker">Query</p>'
            f'<h3>{title}</h3>' + "".join(f"<p>{x}</p>" for x in lines) + "</div></div>")


def fit_of(query_frag, df):
    best = df.iloc[0]
    R_, t = best.fit
    return transparent(M.render_fit(query_frag, M.frags[int(best.other)], R_, t))


def run_index(q, mode):
    df = M.rank_gallery_query(q, MODES[mode])
    partners = M.partners[q]
    ranks = [i + 1 for i, o in enumerate(df.other) if o in partners]
    info = [f"Expert-confirmed partners: <b>{', '.join(NAMES[p] for p in sorted(partners))}</b>",
            f"First true partner ranked <b>#{ranks[0]}</b> of {len(df)} by <b>{mode.split(':')[0].lower()}</b>."]
    return (query_html(cut_uri(q, 320), NAMES[q], info), fit_of(M.frags[q], df),
            candidates_html(df, partners), evidence_html(df, partners))


def on_pick(mode, evt: gr.SelectData):
    return (evt.index,) + run_index(evt.index, mode)


def on_mode(q, mode):
    return run_index(q, mode)


def on_upload(path, mode):
    if path is None:
        raise gr.Error("Upload a sherd image first: a PNG with a transparent background, or a photo on a plain backdrop.")
    fq, df = M.rank_new_image(path, MODES[mode])
    info = ["Uploaded sherd compared with the 124 gold-standard sherds.",
            f"Ranked by <b>{mode.split(':')[0].lower()}</b>. There is no ground truth for uploads."]
    return (query_html(data_uri(cutout(fq), max_side=320), "Your sherd", info), fit_of(fq, df),
            candidates_html(df), evidence_html(df))


PICKER = [(cutout(f), NAMES[i]) for i, f in enumerate(M.frags)]
START = idx["JDX-103"]

# ------------------------------------------------------------------ style
CSS = """
:root { --bg:#15120f; --panel:#211c18; --panel2:#2a241f; --line:#3a322b; --text:#f2ede8; --muted:#b3a597;
        --accent:#d4703c; --accent2:#e8a172; --good:#8fc099; }
body, .gradio-container { background: var(--bg) !important; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
.sm-page, .sm-page * { font-family: 'IBM Plex Sans', system-ui, sans-serif; color: var(--text); }
.sm-page h1, .sm-page h2, .sm-page h3, .sm-page h4 { font-family: 'Fraunces', Georgia, serif; font-weight: 600;
        letter-spacing: -0.01em; margin: 0; }
.sm-kicker { color: var(--accent) !important; text-transform: uppercase; letter-spacing: .14em; font-size: 12px;
        font-weight: 600; margin: 0 0 10px; }
.sm-hero { display: grid; grid-template-columns: 1.05fr 1fr; gap: 32px; align-items: center; min-height: 520px;
        padding: 40px 8px 24px; }
.sm-hero h1 { font-size: clamp(56px, 9vw, 104px); line-height: .95; }
.sm-hero h1 span { color: var(--accent); }
.sm-sub { font-family: 'Fraunces', serif; font-size: 24px; margin: 14px 0 18px; color: var(--accent2) !important; }
.sm-lead { color: var(--muted) !important; font-size: 17px; line-height: 1.6; max-width: 720px; margin: 8px 0 18px; }
.sm-lead em, .sm-card em { color: var(--text); font-style: normal; border-bottom: 1px solid var(--accent); }
.sm-cta { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 22px; }
.sm-btn { background: var(--accent); color: #1b1612 !important; padding: 12px 22px; border-radius: 999px;
        text-decoration: none; font-weight: 600; }
.sm-btn.ghost { background: transparent; color: var(--text) !important; border: 1px solid var(--line); }
.sm-btn:hover { filter: brightness(1.08); }
.sm-hero-art { position: relative; height: 460px; }
.sm-float { position: absolute; rotate: var(--r); filter: drop-shadow(0 18px 24px rgba(0,0,0,.55));
        translate: calc(var(--mx, 0) * var(--z) * 1px) calc(var(--my, 0) * var(--z) * 1px);
        transition: rotate .5s var(--ease), scale .5s var(--ease), translate .25s ease-out;
        animation: sm-fly 1.4s var(--ease) var(--d) backwards, sm-bob 7s ease-in-out calc(var(--d) + 1.4s) infinite; }
.sm-float:nth-child(2n) { animation-duration: 1.4s, 8.5s; }
.sm-float:hover { rotate: 0deg; scale: 1.06; z-index: 3; }
.sm-section { padding: 36px 8px 8px; }
.sm-section.first { padding-top: 24px; }
.sm-section h2 { font-size: 34px; margin-bottom: 20px; }
.sm-h3 { font-size: 24px; margin-bottom: 8px !important; }
.sm-grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.sm-grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
.sm-grid2.wide { grid-template-columns: 1.1fr 1fr; align-items: start; }
.sm-card, .sm-note { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 22px; }
.sm-card.accent { border-color: var(--accent); }
.sm-card p, .sm-note p, .sm-note li { color: var(--muted) !important; line-height: 1.55; margin: 6px 0 0; }
.sm-note b, .sm-card b { color: var(--text); }
.sm-note h3 { font-size: 20px; margin-bottom: 6px; }
.sm-note.warn { border-color: #8a5a3c; }
.sm-note ul { margin: 6px 0 0; padding-left: 18px; }
.sm-big { font-family: 'Fraunces', serif; font-size: 44px; font-weight: 600; color: var(--accent) !important;
        line-height: 1.1; }
.sm-steps { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.sm-steps li { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 20px; }
.sm-steps span, .sm-stephead span { font-family: 'IBM Plex Mono', monospace; color: var(--accent) !important;
        font-size: 14px; }
.sm-steps h4 { font-size: 20px; margin: 8px 0 6px; }
.sm-steps p { color: var(--muted) !important; margin: 0; line-height: 1.5; }
.sm-stephead { display: grid; grid-template-columns: 56px 1fr; gap: 8px; margin-bottom: 14px; }
.sm-stephead span { font-size: 28px; }
.sm-stephead h3 { font-size: 26px; margin-bottom: 6px; }
.sm-stephead p { color: var(--muted) !important; line-height: 1.6; margin: 0; max-width: 900px; }
.sm-fig { margin: 8px 0 0; background: #fff; border-radius: 14px; padding: 12px; }
.sm-fig img { width: 100%; display: block; border-radius: 6px; }
.sm-foot { color: var(--muted) !important; font-size: 13px; margin-top: 28px; padding-top: 16px;
        border-top: 1px solid var(--line); }
.sm-foot a { color: var(--accent2) !important; }
.sm-table { width: 100%; border-collapse: collapse; font-size: 15px; border: none !important; }
.sm-table th, .sm-table td { border-left: none !important; border-right: none !important; border-top: none !important;
        background: transparent !important; }
.sm-table th { text-align: left; color: var(--muted) !important; font-weight: 500; border-bottom: 1px solid var(--line);
        padding: 8px 10px; }
.sm-table td { padding: 9px 10px; border-bottom: 1px solid #2c2620; font-family: 'IBM Plex Mono', monospace; font-size: 14px; }
.sm-table td:first-child { font-family: 'IBM Plex Sans', sans-serif; }
.sm-table tr.hl td { color: var(--accent2) !important; font-weight: 600; }
.sm-table.small td, .sm-table.small th { font-size: 13px; padding: 7px 8px; }
.sm-query { display: flex; gap: 22px; align-items: center; background: var(--panel); border: 1px solid var(--line);
        border-radius: 16px; padding: 18px 22px; }
.sm-query img { width: 150px; height: 150px; object-fit: contain; filter: drop-shadow(0 10px 14px rgba(0,0,0,.5)); }
.sm-query h3 { font-size: 30px; margin: 0 0 6px; }
.sm-query p { color: var(--muted) !important; margin: 3px 0; }
.sm-query b { color: var(--text); }
.sm-cands { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
.sm-cand { position: relative; background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
        padding: 14px 12px 12px; text-align: center; }
.sm-cand.hit { border-color: var(--good); box-shadow: 0 0 0 1px var(--good) inset; }
.sm-cand img { height: 118px; max-width: 100%; object-fit: contain; filter: drop-shadow(0 8px 10px rgba(0,0,0,.5)); }
.sm-rank { position: absolute; top: 8px; left: 10px; font-family: 'IBM Plex Mono', monospace; color: var(--accent) !important; }
.sm-cname { font-weight: 600; margin-top: 6px; }
.sm-bar { height: 6px; background: var(--panel2); border-radius: 3px; margin: 8px 4px 4px; overflow: hidden; }
.sm-bar i { display: block; height: 100%; background: var(--accent); }
.sm-score { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--muted) !important; }
.sm-badge { margin-top: 6px; font-size: 11px; color: var(--good) !important; font-weight: 600; }
.sm-label { color: var(--muted) !important; font-size: 13px; margin: 4px 0 8px !important; }
#sm-picker .grid-wrap { max-height: 430px; overflow-x: hidden; }
#sm-picker .thumbnail-item { background: var(--panel2) !important; border-color: var(--line) !important; }
#sm-picker .thumbnail-item.selected { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent); }
#sm-fit { background: var(--panel) !important; border-radius: 16px; }

/* ---------------- motion ---------------- */
:root { --ease: cubic-bezier(.2, .8, .2, 1); }
@keyframes sm-fly { from { opacity: 0; translate: var(--dx) var(--dy); rotate: var(--r0); scale: .7; } }
@keyframes sm-bob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }
@keyframes sm-up { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: none; } }
@keyframes sm-shine { to { background-position: 200% center; } }
@keyframes sm-glow { 0%, 100% { transform: translate(-10%, -6%) scale(1); opacity: .55; }
                     50% { transform: translate(8%, 6%) scale(1.15); opacity: .85; } }
@keyframes sm-deal { from { opacity: 0; transform: translateY(26px) rotate(-3deg) scale(.94); } }
@keyframes sm-grow { from { transform: scaleX(0); } }
@keyframes sm-pop { from { opacity: 0; transform: scale(.92); filter: blur(4px); } }
@keyframes sm-pulse { 0% { box-shadow: 0 0 0 0 rgba(143,192,153,.55); } 100% { box-shadow: 0 0 0 14px rgba(143,192,153,0); } }

.sm-hero { position: relative; isolation: isolate; }
.sm-hero::before { content: ""; position: absolute; inset: -10% -5% auto 35%; height: 520px; z-index: -1;
        background: radial-gradient(closest-side, rgba(212,112,60,.28), transparent 70%);
        animation: sm-glow 14s ease-in-out infinite; pointer-events: none; }
.sm-hero-text > * { animation: sm-up .9s var(--ease) both; }
.sm-hero-text > :nth-child(2) { animation-delay: .1s; }
.sm-hero-text > :nth-child(3) { animation-delay: .25s; }
.sm-hero-text > :nth-child(4) { animation-delay: .4s; }
.sm-hero-text > :nth-child(5) { animation-delay: .55s; }
.sm-hero h1 span { background-image: linear-gradient(90deg, #d4703c 0%, #f2b483 25%, #d4703c 50%,
        #f2b483 75%, #d4703c 100%) !important; background-size: 200% auto !important; -webkit-background-clip: text;
        background-clip: text; color: transparent !important; animation: sm-shine 6s linear infinite; }

.sm-btn { position: relative; overflow: hidden; transition: transform .25s var(--ease), box-shadow .25s; }
.sm-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(212,112,60,.35); }
.sm-btn::after { content: ""; position: absolute; inset: 0; transform: translateX(-120%);
        background: linear-gradient(100deg, transparent 30%, rgba(255,255,255,.35) 50%, transparent 70%);
        transition: transform .7s var(--ease); }
.sm-btn:hover::after { transform: translateX(120%); }

.sm-card, .sm-note, .sm-steps li { transition: transform .35s var(--ease), border-color .35s, box-shadow .35s,
        opacity .8s var(--ease); }
.sm-card:hover, .sm-note:hover, .sm-steps li:hover { transform: translateY(-4px); border-color: #6a4a35;
        box-shadow: 0 14px 30px rgba(0,0,0,.35); }
.sm-fig { transition: transform .5s var(--ease), box-shadow .5s, opacity .8s var(--ease); }
.sm-fig:hover { transform: scale(1.01); box-shadow: 0 18px 40px rgba(0,0,0,.45); }
.sm-table tbody tr td { transition: background .2s; }
.sm-table tbody tr:hover td { background: rgba(212,112,60,.08) !important; }

/* scroll reveal: JS adds .sm-r to blocks below the fold, then .sm-in when they scroll into view */
.sm-r { opacity: 0; transform: translateY(28px); transition: opacity .8s var(--ease), transform .8s var(--ease) !important;
        transition-delay: var(--sd, 0s) !important; }
.sm-r.sm-in { opacity: 1; transform: none; }

/* matcher: results replay these each time they are re-rendered */
.sm-query { animation: sm-up .6s var(--ease) both; }
.sm-query img { animation: sm-pop .7s var(--ease) both; }
.sm-cand { animation: sm-deal .6s var(--ease) both; transition: transform .3s var(--ease), border-color .3s; }
.sm-cand:nth-child(2) { animation-delay: .08s; } .sm-cand:nth-child(3) { animation-delay: .16s; }
.sm-cand:nth-child(4) { animation-delay: .24s; } .sm-cand:nth-child(5) { animation-delay: .32s; }
.sm-cand:hover { transform: translateY(-6px); }
.sm-cand img { transition: transform .5s var(--ease); }
.sm-cand:hover img { transform: rotate(-4deg) scale(1.08); }
.sm-cand.hit { animation-name: sm-deal, sm-pulse; animation-duration: .6s, 1.4s;
        animation-timing-function: var(--ease), ease-out; animation-iteration-count: 1, 2; animation-fill-mode: both, none; }
.sm-cand.hit:nth-child(1) { animation-delay: 0s, .9s; } .sm-cand.hit:nth-child(2) { animation-delay: .08s, .9s; }
.sm-cand.hit:nth-child(3) { animation-delay: .16s, .9s; } .sm-cand.hit:nth-child(4) { animation-delay: .24s, .9s; }
.sm-cand.hit:nth-child(5) { animation-delay: .32s, .9s; }
.sm-bar i { transform-origin: left; animation: sm-grow 1s var(--ease) .35s both; }
.sm-table.small tbody tr { animation: sm-up .5s var(--ease) both; }
.sm-table.small tbody tr:nth-child(2) { animation-delay: .06s; } .sm-table.small tbody tr:nth-child(3) { animation-delay: .12s; }
.sm-table.small tbody tr:nth-child(4) { animation-delay: .18s; } .sm-table.small tbody tr:nth-child(5) { animation-delay: .24s; }
#sm-fit img.sm-popin { animation: sm-pop .7s var(--ease) both; }
#sm-picker .thumbnail-item { transition: transform .25s var(--ease), border-color .25s !important; }
#sm-picker .thumbnail-item:hover { transform: translateY(-3px) scale(1.03); }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .sm-r { opacity: 1; transform: none; }
}

@media (max-width: 860px) {
  .sm-hero, .sm-grid3, .sm-grid2, .sm-grid2.wide, .sm-steps, .sm-cands { grid-template-columns: 1fr; }
  .sm-hero-art { height: 320px; }
}
"""
MOTION_JS = (Path(__file__).resolve().parent / "assets" / "motion.js").read_text(encoding="utf-8")
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&'
        'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">' + MOTION_JS)


def theme():
    vals = dict(body_background_fill="#15120f", body_text_color="#f2ede8", body_text_color_subdued="#b3a597",
                background_fill_primary="#15120f", background_fill_secondary="#211c18",
                block_background_fill="#211c18", block_border_color="#3a322b", block_label_background_fill="#2a241f",
                block_label_text_color="#b3a597", block_title_text_color="#f2ede8", border_color_primary="#3a322b",
                input_background_fill="#2a241f", input_border_color="#3a322b", panel_background_fill="#211c18",
                button_primary_background_fill="#d4703c", button_primary_background_fill_hover="#e07f4b",
                button_primary_text_color="#1b1612", button_secondary_background_fill="#2a241f",
                button_secondary_text_color="#f2ede8", color_accent="#d4703c", color_accent_soft="#3a2a20",
                checkbox_label_background_fill="#2a241f", checkbox_label_background_fill_selected="#3a2a20",
                checkbox_label_text_color="#f2ede8", checkbox_label_text_color_selected="#f2ede8",
                radio_circle="#d4703c", link_text_color="#e8a172", table_even_background_fill="#211c18",
                table_odd_background_fill="#26201b")
    import inspect
    known = inspect.signature(gr.themes.Base.set).parameters
    vals.update({k + "_dark": v for k, v in list(vals.items()) if k + "_dark" in known})  # same look in dark mode
    return gr.themes.Base(primary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.stone,
                          font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
                          radius_size=gr.themes.sizes.radius_lg).set(**vals)


# ------------------------------------------------------------------ pages
with gr.Blocks(title="SherdMatch") as demo:
    gr.Navbar(main_page_name="Home")
    gr.HTML(HOME, elem_classes="sm-page", padding=False)

with demo.route("How it works", "/method"):
    gr.HTML(METHOD, elem_classes="sm-page", padding=False)

with demo.route("Matcher", "/match"):
    gr.HTML('<div class="sm-page"><section class="sm-section first"><p class="sm-kicker">Matcher</p>'
            '<h2>Find the pieces that join</h2><p class="sm-lead">Click any sherd in the collection, or upload your '
            'own. SherdMatch ranks the most likely joining pieces and previews the best fit. These are suggestions '
            'for an expert to check, not confirmed joins. The demo model is trained on all 111 expert pairs, so '
            'scores for collection sherds are optimistic; the reported results use held-out vessels.</p>'
            '</section></div>', padding=False)
    q_idx = gr.State(START)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=320):
            mode = gr.Radio(list(MODES), value=list(MODES)[0], label="Matching method")
            with gr.Tabs():
                with gr.Tab("Pick from the collection"):
                    gr.HTML('<p class="sm-label">124 gold-standard sherds from Daxinzhuang. Click one.</p>',
                            elem_classes="sm-page", padding=False)
                    picker = gr.Gallery(PICKER, columns=4, height=430, allow_preview=False, show_label=False,
                                        object_fit="contain", elem_id="sm-picker", buttons=[])
                with gr.Tab("Upload a sherd"):
                    up = gr.Image(type="filepath", label="PNG with transparent background, or a photo on a plain backdrop",
                                  height=300)
                    go = gr.Button("Find matches", variant="primary")
        with gr.Column(scale=7, min_width=420):
            qbox = gr.HTML(elem_classes="sm-page", padding=False)
            fit = gr.Image(label="Best fit: digital reconstruction preview (query outlined in dark)", height=330,
                           elem_id="sm-fit", buttons=["download", "fullscreen"], type="numpy")
    gr.HTML('<div class="sm-page"><h3 class="sm-h3" style="margin-top:18px">Top-5 candidates</h3></div>', padding=False)
    cands = gr.HTML(elem_classes="sm-page", padding=False)
    gr.HTML('<div class="sm-page"><h3 class="sm-h3" style="margin-top:18px">Why these candidates</h3>'
            '<p class="sm-label">The evidence behind each score: appearance similarity, how much edge touches, and '
            'how much the pieces would overlap.</p></div>', padding=False)
    table = gr.HTML(elem_classes="sm-page", padding=False)
    outs = [qbox, fit, cands, table]
    picker.select(on_pick, [mode], [q_idx] + outs)
    mode.change(on_mode, [q_idx, mode], outs)
    go.click(on_upload, [up, mode], outs)
    demo.load(on_mode, [q_idx, mode], outs)

with demo.route("Results", "/results"):
    gr.HTML(RESULTS, elem_classes="sm-page", padding=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme(), css=CSS, head=HEAD)
