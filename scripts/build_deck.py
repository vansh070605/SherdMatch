"""Review 2 deck (docs/Review2_PPT.pptx), styled like the Review 1 deck:
dark background, terracotta accent, sherd imagery. Numbers come from outputs/*.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pylibs"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"
ASSETS = OUT / "deck_assets"
DEST = ROOT / "docs" / "Review2_PPT.pptx"

R = json.loads((OUT / "results.json").read_text())
DS = json.loads((OUT / "dataset_summary.json").read_text())
EA = json.loads((OUT / "error_analysis.json").read_text())
PH = json.loads((OUT / "results_real_photos.json").read_text()) if (OUT / "results_real_photos.json").exists() else None
M = {m["method"]: m for m in R["methods"]}
B = R["bootstrap"]
BASE, APPLR, PROP = "DINOv2 embedding (cosine)", "Appearance features (LR)", "Proposed: appearance + geometry (LR)"

BG = RGBColor(0x1B, 0x18, 0x16)
PANEL = RGBColor(0x26, 0x21, 0x1E)
ACCENT = RGBColor(0xD4, 0x70, 0x3C)
TEXT = RGBColor(0xF2, 0xED, 0xE8)
MUTED = RGBColor(0xB3, 0xA5, 0x97)
GOOD = RGBColor(0x8F, 0xC0, 0x99)
BAD = RGBColor(0xE0, 0x7A, 0x6A)
FONT = "Arial"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


# ------------------------------------------------------------------ helpers
def cutouts():
    """Transparent, cropped sherd images from the gold standard for decoration."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    names = ["JDX-100", "JDX-103", "JDX-107", "JDX-33", "JDX-12", "JDX-5", "JDX-58", "JDX-73", "JDX-118", "JDX-139"]
    out = {}
    for n in names:
        dst = ASSETS / f"{n}.png"
        if not dst.exists():
            im = Image.open(ROOT / "data" / "raw" / "dataset_gold_standard" / "dataset" / f"{n}_exterior.png").convert("RGBA")
            im = im.crop(im.getbbox())
            im.thumbnail((700, 700))
            im.save(dst)
        out[n] = dst
    return out


SH = cutouts()


def new_slide(notes=""):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(s, x, y, w, h, paras, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=6, name=None):
    """paras: str or list of str / list of [(text, {opts})] runs. '**x**' marks accent-coloured bold."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [paras] if isinstance(paras, str) else paras
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(spacing)
        parts = p.split("**") if isinstance(p, str) else None
        for j, part in enumerate(parts):
            if not part:
                continue
            r = para.add_run()
            r.text = part
            r.font.name = FONT
            r.font.size = Pt(size)
            hi = j % 2 == 1
            r.font.bold = bold or hi
            r.font.color.rgb = ACCENT if hi else color
    return tb


def bullets(s, x, y, w, h, items, size=15, color=TEXT):
    tb = text(s, x, y, w, h, [""], size=size)
    tf = tb.text_frame
    tf.paragraphs[0]._p.getparent().remove(tf.paragraphs[0]._p)
    for it in items:
        para = tf.add_paragraph()
        para.space_after = Pt(8)
        para.level = 0
        pPr = para._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.22)))
        pPr.set("indent", str(-Inches(0.22)))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        clr = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buClr", {})
        srgb = clr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr", {"val": "D4703C"})
        clr.append(srgb)
        pPr.append(clr)
        pPr.append(bu)
        for j, part in enumerate(it.split("**")):
            if not part:
                continue
            r = para.add_run()
            r.text = part
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = j % 2 == 1
            r.font.color.rgb = ACCENT if j % 2 == 1 else color
    return tb


def header(s, kicker, title, accent=None, title_w=8.5):
    text(s, 0.6, 0.45, 8, 0.3, kicker.upper(), size=12, color=ACCENT, bold=True)
    paras = [title] if accent is None else [title, f"**{accent}**"]
    text(s, 0.6, 0.78, title_w, 1.3, paras, size=34, bold=True, spacing=0)


def panel(s, x, y, w, h, fill=PANEL, line=None):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def card(s, x, y, w, h, title, body, size=13, line=ACCENT):
    panel(s, x, y, w, h, line=line)
    text(s, x + 0.2, y + 0.18, w - 0.4, 0.4, title.upper(), size=12, color=ACCENT, bold=True)
    if isinstance(body, list):
        bullets(s, x + 0.2, y + 0.6, w - 0.4, h - 0.7, body, size=size)
    else:
        text(s, x + 0.2, y + 0.6, w - 0.4, h - 0.7, body, size=size)


def stat(s, x, y, w, big, label, color=ACCENT, size=44):
    off = size / 72 * 1.2
    text(s, x, y, w, off, big, size=size, color=color, bold=True)
    text(s, x, y + off + 0.05, w, 0.6, label, size=13, color=MUTED)


def picture(s, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def framed(s, path, x, y, w, h):
    """White card holding a matplotlib figure, image scaled to fit."""
    panel(s, x, y, w, h, fill=RGBColor(0xFF, 0xFF, 0xFF))
    iw, ih = Image.open(path).size
    pad = 0.12
    sc = min((w - 2 * pad) / iw, (h - 2 * pad) / ih)
    pw, ph = iw * sc, ih * sc
    picture(s, path, x + (w - pw) / 2, y + (h - ph) / 2, w=pw)


def style_chart(chart, legend=True, maxv=None, fmt='0.00', size=12):
    chart.font.name = FONT
    chart.font.size = Pt(size)
    chart.font.color.rgb = TEXT
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.color.rgb = TEXT
        chart.legend.font.size = Pt(size)
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0x3A, 0x34, 0x30)
    va.format.line.fill.background()
    va.tick_labels.font.color.rgb = MUTED
    va.tick_labels.font.size = Pt(size - 1)
    if maxv is not None:
        va.maximum_scale = maxv
        va.minimum_scale = 0
    ca = chart.category_axis
    ca.format.line.color.rgb = RGBColor(0x55, 0x4C, 0x45)
    ca.tick_labels.font.color.rgb = TEXT
    ca.tick_labels.font.size = Pt(size)
    plot = chart.plots[0]
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.font.size = Pt(size - 1)
    dl.font.color.rgb = TEXT
    dl.position = XL_LABEL_POSITION.OUTSIDE_END


def pts(comp, met):
    d, lo, hi = B[comp][met]
    return f"{100 * d:+.1f}", f"95% CI {100 * lo:+.1f} to {100 * hi:+.1f}"


# ------------------------------------------------------------------ slides
# 1 title
s = new_slide("Introduce yourself and the project. SherdMatch is the name of the system built for the AI-based pottery "
              "fragment matching topic presented at Review 1. Review 1 was the plan; Review 2 shows what was built, "
              "the results, and an honest answer to the research question.")
text(s, 0.6, 0.8, 6.8, 1.2, "Sherd**Match**", size=66, bold=True, spacing=0)
text(s, 0.6, 2.15, 6.6, 1.4, ["AI-Based Archaeological", "Pottery Fragment Matching"], size=28, bold=True, spacing=0)
text(s, 0.6, 3.75, 6.2, 0.8, "From a plan to a working, evaluated system: appearance + fracture-edge geometry "
                             "on the Daxinzhuang-18K benchmark", size=16, color=MUTED)
text(s, 0.6, 4.75, 6, 0.4, "PRT LLT · 21CSE451T", size=14, color=ACCENT, bold=True)
text(s, 0.6, 5.1, 6, 0.4, "Review 2", size=18, bold=True)
text(s, 0.6, 5.75, 6, 1.1, ["Vansh Agrawal · RA2311026010120", "CINTEL · Section AC-1", "Faculty: Dr Nithyashri J"],
     size=13, color=MUTED, spacing=2)
for n, (x, y, w) in {"JDX-100": (8.0, 0.8, 2.6), "JDX-103": (10.7, 1.3, 2.1), "JDX-33": (7.6, 3.6, 2.2),
                     "JDX-107": (10.0, 3.9, 2.4), "JDX-118": (9.4, 2.3, 1.4)}.items():
    picture(s, SH[n], x, y, w=w)

# 2 what changed since review 1
s = new_slide("Review 1 ended with 'planned methodology, no experimental results yet'. Everything on this slide now "
              "exists and is public on GitHub.")
header(s, "Progress since Review 1", "From plan to evidence")
items = [("Review 1", "Dataset named, schema to verify, pipeline planned, no results", MUTED),
         ("Review 2", "Data acquired and verified, pipeline built, 5 experiments, working demo", ACCENT)]
for i, (t, b_, c) in enumerate(items):
    panel(s, 0.6 + i * 3.1, 2.1, 2.9, 1.5, line=c)
    text(s, 0.8 + i * 3.1, 2.25, 2.5, 0.4, t.upper(), size=12, color=c, bold=True)
    text(s, 0.8 + i * 3.1, 2.7, 2.5, 0.9, b_, size=13)
for i, (big, lab) in enumerate([(str(DS["gold_sherds_exterior"]), "real sherds in the test set"),
                                (str(DS["gold_refit_pairs"]), "expert-confirmed joins"),
                                ("16", "model configurations compared"), ("5", "experiments run (E1–E5)")]):
    stat(s, 0.6 + (i % 2) * 3.1, 4.1 + (i // 2) * 1.55, 2.9, big, lab)
bullets(s, 7.2, 2.1, 5.5, 4.8, [
    "**Data**: Daxinzhuang-18K found on Zenodo; only the needed parts fetched (range requests, CRC-checked)",
    "**Pipeline**: segmentation, CLAHE, contours, colour/LBP/ResNet-50/DINOv2 features",
    "**Geometry**: complementary-curvature edge matcher with Kabsch + ICP alignment",
    "**Model**: logistic-regression fusion, compared with RF and SVM",
    "**Evaluation**: leak-free vessel-grouped CV, bootstrap CIs, ablations",
    "**Prototype**: Gradio demo with ranked candidates and fit preview",
], size=15)

# 3 dataset
s = new_slide("Explain the four parts and what each was used for. Stress the imbalance: 1.46% of pairs are joins, so "
              "accuracy is meaningless and we use ranking metrics.")
header(s, "Dataset", "Daxinzhuang-18K, verified")
rows = [("Real photos", "36,532 photos · 17,635 records", "metadata + photo segmentation"),
        ("Gold standard", "124 sherds · 111 expert joins + poses", "main train / test set"),
        ("Simulated", "50,736 synthetic break pairs", "geometry sanity check (2,212 used)"),
        ("Refit sheet", "251 expert joining groups", "ground truth, raw-photo test (next)")]
for i, (a, b_, c) in enumerate(rows):
    y = 2.15 + i * 0.95
    panel(s, 0.6, y, 7.2, 0.8)
    text(s, 0.8, y + 0.12, 1.8, 0.6, a, size=14, color=ACCENT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 2.65, y + 0.1, 2.9, 0.6, b_, size=13, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 5.6, y + 0.1, 2.1, 0.6, c, size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
stat(s, 8.4, 2.0, 4.4, "1.46%", "of the 7,626 sherd pairs are joins: extreme class imbalance", size=54)
stat(s, 8.4, 3.75, 4.4, "33", f"vessel groups (2–{DS['group_size_max']} sherds each), used to split train/test", size=54)
text(s, 0.6, 6.15, 7.2, 0.6, "Zenodo 21553799 · Ji et al., 2026 · CC-BY 4.0", size=12, color=MUTED)
for k, n in enumerate(["JDX-12", "JDX-73", "JDX-139"]):
    picture(s, SH[n], 8.5 + k * 1.5, 5.6, h=1.2)

# 4 metadata
s = new_slide("Two real pitfalls found in the metadata. The photo-ID bug would have silently paired wrong photos. "
              "The pose conventions differ between gold and simulated data; both were verified by checking that "
              "true partners touch without overlapping.")
header(s, "Metadata understanding", "What pixels cannot", "tell us")
card(s, 0.6, 2.55, 3.9, 2.0, "Photo ID ≠ sherd ID",
     "Sherd JD00001 → photos JD_00001 (exterior) + JD_00002 (interior). Mapping by number would pair the wrong photos. "
     "Fixed via the Exterior_Photo column.", size=13)
card(s, 0.6, 4.75, 3.9, 2.0, "Pose conventions verified",
     "Gold: recentre on alpha centroid, rotate, place. Simulated: image centre, opposite sign. Verified: true pairs "
     "touch 75–170 px with ≤7% overlap.", size=13)
card(s, 4.7, 2.55, 3.0, 4.2, "Used for", ["**Vessel groups** → leak-free folds",
                                         "**Notes** (spalling) explain errors",
                                         "**Layer, part** → future priors",
                                         f"Only {DS['vessel_type_known']} of 17,635 records typed: too sparse as a label"],
     size=13)
framed(s, FIG / "gold_gt_assemblies.png", 7.95, 2.55, 4.8, 2.4)
text(s, 7.95, 5.1, 4.8, 1.2, "Expert joins re-rendered with our own pose code. Visible gaps along the joints are "
                             "edge erosion, the central obstacle for geometry.", size=13, color=MUTED)

# 5 preprocessing
s = new_slide("Walk through the five steps on one sherd. Then the raw photos: the scale card is rejected by a "
              "rectangularity filter, GrabCut refines the mask. Mention the remaining errors honestly.")
header(s, "The process, implemented", "Preprocessing pipeline")
framed(s, FIG / "preprocessing_pipeline.png", 0.6, 2.2, 12.1, 2.5)
framed(s, FIG / "photo_segmentation.png", 0.6, 4.95, 7.2, 2.2)
bullets(s, 8.1, 4.95, 4.6, 2.3, [
    "Scale kept fixed: edge lengths must match",
    "Raw photos: Otsu → card filter → **GrabCut**",
    "Remaining error: light edges cut, shadows kept",
], size=14)

# 6 model
s = new_slide("The selected model is a pairwise classifier. Appearance and geometry features of a pair go into a "
              "class-balanced logistic regression; its probability ranks the gallery.")
header(s, "Model selection", "Two cues, one", "interpretable classifier")
framed(s, FIG / "method_architecture.png", 0.6, 2.75, 7.6, 2.7)
card(s, 8.45, 2.75, 4.3, 1.7, "Pattern to recognise", "Does sherd pair (A, B) join? Used as ranking: sort the "
                                                     "gallery by P(join).", size=13)
card(s, 8.45, 4.6, 4.3, 2.3, "Logistic regression", ["P(join) = σ(wᵀz + b)", "12 standardised pair features",
                                                     "class-balanced, L2 (C = 1)"], size=13)
text(s, 0.6, 5.65, 7.6, 1.2, ["**4 appearance**: colour histogram, LBP texture, ResNet-50, DINOv2",
                              "**8 geometry**: contact, contact fraction, overlap, rmse, fit score, stripe-angle gap "
                              "(+confidence), area ratio"], size=13)

# 7 edge matcher
s = new_slide("Key idea: at a join the two outlines run in opposite directions and A's bumps are B's dents. We compare "
              "A's curvature with B's reversed and negated curvature, align with Kabsch, refine with ICP, and reject "
              "fits that overlap.")
header(s, "Fracture-edge geometry", "Complementary curvature matching", title_w=11)
framed(s, FIG / "method_edge_matcher.png", 0.6, 2.05, 12.1, 3.5)
for i, (t, b_) in enumerate([("01 Signature", "turning angle every 3 px"), ("02 Seeds", "A vs reversed −B windows"),
                             ("03 Align", "Kabsch, then ICP"), ("04 Verify", "contact ↑ overlap ↓ stripes agree")]):
    x = 0.6 + i * 3.1
    text(s, x, 5.85, 2.9, 0.35, t, size=14, color=ACCENT, bold=True)
    text(s, x, 6.25, 2.9, 0.6, b_, size=13)

# 8 justification
s = new_slide("Justification is worth the most marks. LR wins on ranking, is interpretable, and suits 111 positives. "
              "RF has similar F1 but ranks worse; SVM is worst. A deep Siamese network would need thousands of real "
              "joins.")
header(s, "Justification", "Why logistic regression")
cd = CategoryChartData()
cd.categories = ["Hit@5", "Hit@10", "MRR"]
for n, lab in [(BASE, "Image-only (DINOv2)"), ("Appearance + geometry (SVM-RBF)", "SVM-RBF"),
               ("Appearance + geometry (Random Forest)", "Random Forest"), (PROP, "Logistic regression")]:
    cd.add_series(lab, [M[n]["hit@5"], M[n]["hit@10"], M[n]["mrr"]])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(2.1), Inches(7.2), Inches(4.8), cd).chart
style_chart(ch, maxv=0.85)
for ser, col in zip(ch.series, [MUTED, RGBColor(0x6E, 0x63, 0x5A), RGBColor(0xE8, 0xB0, 0x8C), ACCENT]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
card(s, 8.1, 2.1, 4.6, 1.45, "Data size", "111 positives fit a 13-parameter model; non-linear models and deep nets "
                                         "overfit", size=13)
card(s, 8.1, 3.75, 4.6, 1.45, "Interpretability", "Signed weight per cue: experts can see why a sherd was suggested",
     size=13)
card(s, 8.1, 5.4, 4.6, 1.45, "Cost and result", "Milliseconds to train; best Hit@5 and Hit@10 of all classifiers",
     size=13)

# 9 setup
s = new_slide("Leak-free protocol: whole vessels are held out, so the model cannot win by memorising a vessel's colour. "
              "Thresholds are chosen on training folds only. Accuracy is reported but ignored: all-'no join' gives 98.5%.")
header(s, "Experimental setup", "Leak-free evaluation")
card(s, 0.6, 2.1, 4.0, 4.8, "Validation", ["5-fold **GroupKFold** over the 33 vessel groups",
                                            "No vessel in both train and test",
                                            "Each query ranked against all 123 others",
                                            "F1 threshold from inner CV on training pairs"], size=14)
card(s, 4.75, 2.1, 4.0, 4.8, "Metrics and why", ["**Hit@k**: what the expert checks",
                                                  "**MRR, mAP**: rank quality",
                                                  "**P / R / F1, PR-AUC**: pair decisions",
                                                  "Accuracy is misleading at 1.46% positives",
                                                  "**Bootstrap 95% CIs** (5,000×)"], size=14)
card(s, 8.9, 2.1, 3.8, 4.8, "Experiments", ["E1 image-only vs fusion", "E2 LR vs RF vs SVM", "E3 ablation of every cue",
                                             "E4 geometry on synthetic breaks", "E5 pose accuracy on real joins",
                                             "E6 raw photos (next step)"], size=14)

# 10 headline results
s = new_slide(f"Headline: the proposed model finds a true partner in the top 10 for {M[PROP]['hit@10']:.0%} of sherds "
              f"vs {M[BASE]['hit@10']:.0%} for the image-only plan. The gain is significant at top-5 and top-10.")
header(s, "Results (E1)", "Better than image-only retrieval")
cd = CategoryChartData()
cd.categories = ["Hit@1", "Hit@5", "Hit@10", "MRR"]
for n, lab in [("Random ranking", "Random"), (BASE, "Image-only (DINOv2)"), (APPLR, "Appearance (LR)"),
               (PROP, "Proposed: appearance + geometry")]:
    cd.add_series(lab, [M[n]["hit@1"], M[n]["hit@5"], M[n]["hit@10"], M[n]["mrr"]])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(2.1), Inches(8.0), Inches(4.9), cd).chart
style_chart(ch, maxv=0.85)
for ser, col in zip(ch.series, [RGBColor(0x55, 0x4C, 0x45), MUTED, RGBColor(0xE8, 0xB0, 0x8C), ACCENT]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
d10, c10 = pts("proposed_vs_dinov2", "hit@10")
d5, c5 = pts("proposed_vs_dinov2", "hit@5")
stat(s, 9.0, 2.2, 3.8, f"{M[PROP]['hit@10']:.0%}", "of sherds have a true partner in the top 10", size=54)
stat(s, 9.0, 3.85, 3.8, f"{d10} pts", f"Hit@10 vs image-only ({c10})", size=40)
stat(s, 9.0, 5.35, 3.8, f"{d5} pts", f"Hit@5 vs image-only ({c5})", size=40)

# 11 does geometry help
s = new_slide("The honest part. Compared with learned appearance alone, geometry adds a non-significant +3 to +4 points "
              "at top-5/10 and loses 6.5 points at top-1. The ablation shows the orientation cue hurts. We kept the "
              "pre-specified model rather than cherry-picking the best ablation.")
header(s, "Testing the hypothesis", "Does geometry pull its weight?")
labels = {"Ablation: proposed - colour": "− colour", "Ablation: proposed - deep embeddings": "− deep embeddings",
          "Ablation: proposed - overlap check": "− overlap check", "Ablation: proposed - edge contact": "− edge contact",
          "Ablation: proposed - orientation cue": "− orientation cue"}
cd = CategoryChartData()
cd.categories = list(labels.values())
delta = [M[k]["hit@5"] - M[PROP]["hit@5"] for k in labels]
cd.add_series("cue helped", [d if d < 0 else None for d in delta])
cd.add_series("removing it helps", [d if d >= 0 else None for d in delta])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(2.1), Inches(6.4), Inches(4.4), cd)
ch = gf.chart
style_chart(ch, legend=False, fmt='+0.00;-0.00')
ch.plots[0].overlap = 100
ch.plots[0].gap_width = 60
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
ch.has_title = True
ch.chart_title.text_frame.text = "Removing a cue: change in Hit@5"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = TEXT
for ser, col in zip(ch.series, [BAD, GOOD]):
    ser.invert_if_negative = False
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
for i, d in enumerate(delta):  # negative labels inside the bar, away from the category names
    if d < 0:
        dlab = ch.series[0].points[i].data_label
        dlab.position = XL_LABEL_POSITION.INSIDE_END
        dlab.text_frame.text = f"{d:+.2f}"  # per-point labels ignore the series number format
        run = dlab.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT
text(s, 0.6, 6.6, 6.4, 0.6, "Red = the cue helped (removing it hurts). Green = removing it helps.", size=12, color=MUTED)
for i, (met, lab) in enumerate([("hit@1", "Hit@1"), ("hit@5", "Hit@5"), ("hit@10", "Hit@10")]):
    d, c = pts("proposed_vs_appearanceLR", met)
    sig = B["proposed_vs_appearanceLR"][met][1] > 0 or B["proposed_vs_appearanceLR"][met][2] < 0
    y = 2.1 + i * 1.25
    panel(s, 7.35, y, 5.4, 1.05)
    text(s, 7.55, y + 0.12, 1.3, 0.8, lab, size=15, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.8, y + 0.08, 1.6, 0.85, f"{d}", size=26, bold=True, color=BAD if float(d) < 0 else GOOD,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 10.35, y + 0.1, 2.3, 0.85, [c, "significant" if sig else "not significant"], size=11, color=MUTED,
         spacing=0, anchor=MSO_ANCHOR.MIDDLE)
text(s, 7.35, 5.95, 5.4, 1.0, "Adding geometry to learned appearance: **H1 only partly supported**", size=16)

# 12 synthetic vs real
s = new_slide("Why geometry under-delivers: the same matcher recovers about half of clean synthetic breaks but under a "
              "fifth of real ones. At the expert pose real partners touch along 87 px; the best wrong fit touches 129 px. "
              "Maximising contact prefers wrong fits on worn edges.")
header(s, "Diagnosis (E4, E5)", "The bottleneck is the edge,", "not the matcher")
stat(s, 0.6, 2.35, 2.9, f"{R['sim']['pose_success_rate_50px']:.0%}", "clean synthetic breaks aligned within 50 px",
     size=54, color=TEXT)
stat(s, 3.6, 2.35, 2.9, f"{R['gold_pose']['pose_success_rate_50px']:.0%}", "real eroded sherds aligned within 50 px",
     size=54)
bullets(s, 6.9, 2.35, 5.8, 1.9, [
    "At the expert pose, true partners touch along **87 px** (median)",
    "The best **wrong** pose found by ICP touches along **129 px**",
    "On worn edges, maximising contact chooses wrong fits",
], size=15)
framed(s, FIG / "results_alignment_examples.png", 0.6, 4.35, 12.1, 2.85)

# 13 errors
e = EA["prop"]
s = new_slide("Error analysis. Most wrong top-1 picks are look-alike sherds from other vessels of the same ware. "
              "The top-5 still contains 7x more same-vessel sherds than chance. Small sherds are hardest. Pair-level "
              "precision is modest, so this is a ranking aid, not an automatic verdict.")
header(s, "Results analysis", "Where it fails, and why")
framed(s, FIG / "results_qualitative_retrieval.png", 0.6, 2.1, 5.2, 4.9)
stat(s, 6.15, 2.1, 3.2, f"{e['frac']:.0%}", "of wrong top-1 picks are from the same vessel: the confusion is between "
                                            "vessels of one ware", size=40)
stat(s, 9.55, 2.1, 3.2, f"{e['top5_same_vessel_share'] / 0.032:.0f}×", "more same-vessel sherds in the top 5 than "
                                                                        "chance", size=40)
stat(s, 6.15, 3.95, 3.2, f"{e['hit5_by_area_tercile'][0]:.2f}→{e['hit5_by_area_tercile'][2]:.2f}",
     "Hit@5, smallest vs largest third of sherds", size=34)
m = M[PROP]
stat(s, 9.55, 3.95, 3.2, f"{m['pair_pr_auc'] / 0.072:.1f}×", f"PR-AUC over chance (precision {m['pair_precision']:.2f}, "
                                                             f"recall {m['pair_recall']:.2f}, F1 {m['pair_f1']:.2f})",
     size=40)
triv = (m["pair_tn"] + m["pair_fp"]) / (m["pair_tn"] + m["pair_fp"] + m["pair_tp"] + m["pair_fn"])
text(s, 6.15, 5.85, 6.6, 1.0, f"Accuracy is {m['pair_accuracy']:.1%}, but predicting 'no join' for everything already "
                              f"scores {triv:.1%} on these test pairs, so it is not a useful measure here.", size=13, color=MUTED)

# 14 E6 raw photos (if available)
if PH:
    s = new_slide("Zero-shot test on raw field photos: automatic segmentation, models trained only on the gold set, "
                  f"{PH['n_photos']} photos in the gallery. Shows whether conclusions carry over to real use.")
    header(s, "Generalisation (E6)", "Raw excavation photos, zero-shot")
    ph = PH["methods"]
    keys = [("Random ranking", "Random"), ("DINOv2 embedding (cosine)", "Image-only (DINOv2)"),
            ("Colour histogram (chi2)", "Colour histogram"), ("Appearance features (LR, trained on gold)", "Appearance (LR)"),
            ("Proposed two-stage: appearance top-30 + geometry re-rank", "Two-stage + geometry")]
    cd = CategoryChartData()
    cd.categories = ["Hit@1", "Hit@5", "Hit@10", "MRR"]
    for k, lab in keys:
        cd.add_series(lab, [ph[k]["hit@1"], ph[k]["hit@5"], ph[k]["hit@10"], ph[k]["mrr"]])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(2.1), Inches(8.0), Inches(4.9), cd).chart
    style_chart(ch, maxv=max(0.5, max(ph[k]["hit@10"] for k, _ in keys) + 0.1), size=11)
    for ser, col in zip(ch.series, [RGBColor(0x55, 0x4C, 0x45), MUTED, RGBColor(0x6E, 0x63, 0x5A),
                                    RGBColor(0xE8, 0xB0, 0x8C), ACCENT]):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
    stat(s, 9.0, 2.2, 3.8, str(PH["n_photos"]), "photos in the gallery (about 6× the gold set)", size=48)
    stat(s, 9.0, 3.9, 3.8, str(PH["n_queries"]), "queries with an expert-confirmed partner", size=48)
    text(s, 9.0, 5.6, 3.8, 1.3, "Harder than the benchmark: automatic segmentation, varied lighting and backdrops.",
         size=13, color=MUTED)

# 15 demo
s = new_slide("Live demo: pick JDX-103; the expert partner JDX-107 comes first and the fit preview shows the joint. "
              "Switch to the image-only baseline to show the difference. Upload a raw photo to show segmentation.")
header(s, "Prototype", "Working demo")
framed(s, FIG / "demo_gallery_query.jpg", 0.6, 2.1, 7.4, 4.9)
bullets(s, 8.3, 2.2, 4.4, 3.4, [
    "Pick a sherd or **upload a photo**",
    "Top-5 candidates with scores",
    "**Evidence table**: why each was suggested",
    "**Fit preview**: first step to digital reconstruction",
    "Switch: proposed vs image-only",
], size=15)
panel(s, 8.3, 5.75, 4.4, 1.2, line=ACCENT)
text(s, 8.5, 5.88, 4.0, 1.0, ["python app.py", "github.com/vansh070605/SherdMatch"], size=13,
     color=TEXT, spacing=4)

# 16 inference
s = new_slide("Summarise the inference: practical triage tool, hypothesis partly supported, geometry limited by edge "
              "erosion, colour is the strongest cue. Mention the claims we do not make.")
header(s, "Inference", "What the evidence supports")
for i, (t, b_) in enumerate([
    ("Useful triage tool", f"An expert checking 10 suggestions finds a partner for {M[PROP]['hit@10']:.0%} of sherds"),
    ("H1 partly supported", "Better than image-only, but the gain comes from fusing appearance cues"),
    ("Edges are the bottleneck", "2-D outline geometry fails on eroded edges, not on clean ones"),
    ("Colour matters most", "Vessels fired unevenly: colour identifies the vessel better than shape"),
]):
    x, y = 0.6 + (i % 2) * 4.0, 2.1 + (i // 2) * 2.45
    card(s, x, y, 3.8, 2.25, t, b_, size=14)
card(s, 8.75, 2.1, 4.0, 4.6, "Claims we do not make", ["Not an automatic reconstruction",
                                                         "Not 'geometry is useless': 3-D and thickness untested",
                                                         "One site, one ware"], size=14, line=MUTED)

# 17 limitations + future
s = new_slide("Limitations and realistic next steps, ordered by effort.")
header(s, "Limitations & future scope", "What comes next")
card(s, 0.6, 2.1, 5.6, 4.8, "Limitations", ["2-D outlines miss thickness and wall curvature",
                                            "Max-contact pose choice fails on worn edges",
                                            "Only 111 positives: wide confidence intervals",
                                            "Exterior view only; one site and ware",
                                            "Raw-photo masks: shadows, light edges"], size=14, line=MUTED)
for i, (t, b_, eff) in enumerate([("Interior view + edge-band colour", "independent evidence across the joint", "LOW"),
                                  ("Learned pose scoring", "train on expert vs false ICP poses", "MEDIUM"),
                                  ("Contrastive fine-tuning", "better vessel grouping", "MEDIUM"),
                                  ("Multi-piece assembly", "erosion-aware simulation → full reconstruction",
                                   "HIGH")]):
    y = 2.1 + i * 1.22
    panel(s, 6.45, y, 6.3, 1.05)
    text(s, 6.65, y + 0.13, 4.4, 0.4, t, size=14, bold=True)
    text(s, 6.65, y + 0.55, 4.4, 0.4, b_, size=12, color=MUTED)
    text(s, 11.2, y + 0.3, 1.4, 0.4, eff, size=12, color=ACCENT, bold=True, align=PP_ALIGN.RIGHT)

# 18 references + close
s = new_slide("Close and invite questions.")
header(s, "References", "Sources and code")
refs = [
    "Ji, D., et al. (2026). Daxinzhuang-18K: A large-scale 2D benchmark for pottery sherd retrieval and reassembly. Zenodo. doi:10.5281/zenodo.21553799",
    "Freeman, H., & Garder, L. (1964). Apictorial jigsaw puzzles. IEEE Trans. Electronic Computers, EC-13(2), 118–127.",
    "Huang, Q.-X., et al. (2006). Reassembling fractured objects by geometric matching. ACM TOG, 25(3), 569–578.",
    "Pinho, M., Mota, G., & da Costa, G. (2025). A deep learning approach to assist in pottery reconstruction from its sherds. Heritage, 8(5), 167.",
    "Yu, X., et al. (2026). Mitigating spurious features by contrastive learning in pottery sherd recognition. npj Heritage Science, 14, 135.",
    "Oquab, M., et al. (2024). DINOv2: Learning robust visual features without supervision. TMLR.",
    "He, K., et al. (2016). Deep residual learning for image recognition. CVPR.",
    "Kabsch, W. (1976). Acta Cryst. A32, 922–923 · Besl, P., & McKay, N. (1992). IEEE TPAMI 14(2), 239–256.",
    "Full list (23 references): Review 2 case study report, Section 16.",
]
text(s, 0.6, 2.1, 8.2, 4.8, refs, size=12, color=TEXT, spacing=7)
panel(s, 9.2, 2.1, 3.55, 2.2, line=ACCENT)
text(s, 9.4, 2.3, 3.2, 1.9, ["**Code, results, report**", "github.com/vansh070605/", "SherdMatch"],
     size=14, spacing=4)
text(s, 9.2, 4.7, 3.6, 0.8, "Thank you", size=34, bold=True)
text(s, 9.2, 5.45, 3.6, 0.5, "Questions?", size=20, color=ACCENT, bold=True)
picture(s, SH["JDX-107"], 11.4, 5.75, h=1.4)

DEST.parent.mkdir(exist_ok=True)
prs.save(str(DEST))
print("saved", DEST, len(prs.slides), "slides")
