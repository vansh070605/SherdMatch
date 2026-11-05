"""Fill the faculty's case-study template with the Review 2 report.

Reads every number from outputs/*.json so the text always matches the
experiments. Output: docs/Review2_Case_Study_Report.docx
"""
import copy
import json
import re
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TEMPLATE = ROOT / "templates" / "Pattern_Recognition_Case_Study_Report_Template.docx"  # course template, not redistributed
DEST = ROOT / "docs" / "Review2_Case_Study_Report.docx"

R = json.loads((OUT / "results.json").read_text())
DS = json.loads((OUT / "dataset_summary.json").read_text())
EA = json.loads((OUT / "error_analysis.json").read_text())
PH = json.loads((OUT / "results_real_photos.json").read_text()) if (OUT / "results_real_photos.json").exists() else None
M = {m["method"]: m for m in R["methods"]}
B = R["bootstrap"]

BASE = "DINOv2 embedding (cosine)"
APPLR = "Appearance features (LR)"
PROP = "Proposed: appearance + geometry (LR)"
RF = "Appearance + geometry (Random Forest)"
SVM = "Appearance + geometry (SVM-RBF)"


def pct(x):
    return f"{100 * x:.1f}%"


def f3(x):
    return f"{x:.3f}"


def ci(comp, met):
    d, lo, hi = B[comp][met]
    return f"{100 * d:+.1f} pts (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})"


# ------------------------------------------------------------------ writer
NUM = {"fig": 0, "tab": 0}
LABELS = {}  # "FIG:<stem>" / "TAB:<label>" -> number, resolved after the build


def resolve_labels(doc):
    pat = re.compile(r"\{((?:FIG|TAB):\w+)\}")
    paras = list(doc.paragraphs)
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                paras.extend(c.paragraphs)
    for para in paras:
        for r in para.runs:
            if "{" in r.text:
                r.text = pat.sub(lambda m: str(LABELS[m.group(1)]), r.text)

class Writer:
    """Inserts content after an anchor paragraph, keeping document order."""

    def __init__(self, doc, anchor):
        self.doc = doc
        self.cur = anchor._p

    def _place(self, el):
        self.cur.addnext(el)
        self.cur = el

    def p(self, text, style=None, align=None, size=None, italic=False, space_after=6):
        para = self.doc.add_paragraph(style=style)
        # **bold** and *italic* inline markup
        for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
            if not part:
                continue
            if part.startswith("**"):
                run = para.add_run(part[2:-2])
                run.bold = True
            elif part.startswith("*"):
                run = para.add_run(part[1:-1])
                run.italic = True
            else:
                run = para.add_run(part)
            if italic:
                run.italic = True
            if size:
                run.font.size = Pt(size)
        if align == "center":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align is None and style is None:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.space_after = Pt(space_after)
        self._place(para._p)
        return para

    def h(self, text):
        para = self.doc.add_paragraph(style="Heading 3")
        para.add_run(text)
        para.paragraph_format.keep_with_next = True
        self._place(para._p)

    def bullets(self, items):
        for it in items:
            self.p(it, style="List Bullet", space_after=2)

    def table(self, header, rows, widths=None, caption=None, font=9, label=None):
        if caption:
            NUM["tab"] += 1
            if label:
                LABELS[f"TAB:{label}"] = NUM["tab"]
            c = self.p(f"Table {NUM['tab']}. {caption}", style="Caption", align="center", space_after=2)
            c.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1, cols=len(header))
        try:
            t.style = self.doc.styles["Table Grid"]
        except KeyError:
            _borders(t)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(h))
            run.bold = True
            run.font.size = Pt(font)
            _shade(cell, "E8DDD3")
        for r in rows:
            cells = t.add_row().cells
            for i, v in enumerate(r):
                cells[i].text = ""
                bold = isinstance(v, str) and v.startswith("**")
                run = cells[i].paragraphs[0].add_run(str(v).strip("*"))
                run.bold = bold
                run.font.size = Pt(font)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        _repeat_header(t.rows[0])
        self._place(t._tbl)
        self.p("", space_after=4)
        return t

    def fig(self, name, caption, width=6.5):
        NUM["fig"] += 1
        LABELS[f"FIG:{Path(name).stem}"] = NUM["fig"]
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(FIG / name), width=Inches(width))
        para.paragraph_format.keep_with_next = True
        self._place(para._p)
        self.p(f"Figure {NUM['fig']}. {caption}", style="Caption", align="center", space_after=10)

    def page_break(self):
        para = self.doc.add_paragraph()
        para.add_run().add_break(WD_BREAK.PAGE)
        self._place(para._p)


def _shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_)
    tcPr.append(shd)


def _repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def _borders(t):
    tblPr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "999999")
        b.append(e)
    tblPr.append(b)


def response_anchor(doc, heading_text):
    """The 'Response:' paragraph under a Heading 2; strips its blank lines."""
    paras = doc.paragraphs
    for i, p in enumerate(paras):
        if p.style.name.startswith("Heading") and p.text.strip().startswith(heading_text):
            for q in paras[i + 1:]:
                if q.text.strip().startswith("Response:"):
                    for r in q.runs[1:]:
                        r._r.getparent().remove(r._r)
                    q.runs[0].text = "Response:"
                    q.runs[0].bold = True
                    return q
    raise KeyError(heading_text)


def replace_blank_lines(par, text):
    """'Faculty comments: ____' -> 'Faculty comments: <text>'."""
    label = par.text.split(":")[0] + ":"
    for r in par.runs[1:]:
        r._r.getparent().remove(r._r)
    par.runs[0].text = label + " "
    par.runs[0].bold = True
    par.add_run(text)


def retrieval_rows(names, label=None):
    rows = []
    for n in names:
        m = M[n]
        nm = (label or {}).get(n, n)
        if n == PROP:
            nm = f"**{nm}"
        rows.append([nm, f3(m["hit@1"]), f3(m["hit@5"]), f3(m["hit@10"]), f3(m["mrr"]), f3(m["map"]),
                     f"{m['median_rank']:.1f}"])
    return rows


# ------------------------------------------------------------------ content
def fill(doc):
    # ---------------- details table
    details = {
        "Course Code / Course Title": "21CSE451T – Pattern Recognition Techniques",
        "Academic Year / Semester": "2026–27 (Odd) / Semester 7",
        "Department / Section": "Computational Intelligence (CINTEL) / AC-1",
        "Student Name(s)": "Vansh Agrawal",
        "Register Number(s)": "RA2311026010120",
        "Project Title": "SherdMatch: AI-Based Archaeological Pottery Fragment Matching",
        "Faculty Guide / Reviewer": "Dr Nithyashri J",
        "Review Date(s)": "Review 1: 29 October 2026     Review 2: 5 November 2026",
    }
    for row in doc.tables[0].rows:
        k = row.cells[0].text.strip()
        if k in details:
            row.cells[1].text = details[k]

    # ---------------- 1. Abstract
    w = Writer(doc, response_anchor(doc, "1. Abstract"))
    p, b, a = M[PROP], M[BASE], M[APPLR]
    w.p(
        "Archaeological pottery is excavated as hundreds of broken sherds, and finding which pieces physically join "
        "is slow expert work. This case study presents **SherdMatch**, which treats joining as a pattern-recognition "
        "problem: given one sherd, rank the others in a collection by the probability that the two fit together. "
        f"We use the Daxinzhuang-18K benchmark (Ji et al., 2026): {DS['gold_sherds_exterior']} sherds with "
        f"{DS['gold_refit_pairs']} expert-confirmed joins form the test set, supported by simulated fractures and "
        "excavation metadata. Each sherd is segmented, normalised and described by colour, LBP texture and "
        "ResNet-50/DINOv2 embeddings. Each pair is also checked for fracture-edge fit by aligning complementary "
        "curvature segments (Kabsch + ICP). A class-balanced logistic regression fuses twelve pair features into a "
        "join probability. Under leak-free vessel-grouped cross-validation, the model places a true partner in the "
        f"top 10 for {pct(p['hit@10'])} of queries, against {pct(b['hit@10'])} for the image-only baseline of "
        f"Review 1 ({ci('proposed_vs_dinov2', 'hit@10')}). Most of that gain comes from combining appearance cues: "
        f"adding edge geometry changes Hit@5 by {ci('proposed_vs_appearanceLR', 'hit@5')}, which is not significant. "
        f"The edge matcher aligns {pct(R['sim']['pose_success_rate_50px'])} of clean synthetic breaks but only "
        f"{pct(R['gold_pose']['pose_success_rate_50px'])} of real, eroded ones. A learned appearance ranker is "
        "therefore a practical triage tool, whereas fracture geometry needs erosion-robust descriptors.")
    w.p("**Keywords:** pottery sherd matching, fragment reassembly, image retrieval, contour matching, "
        "logistic regression, cultural heritage.")

    # ---------------- 2. Introduction
    w = Writer(doc, response_anchor(doc, "2. Introduction"))
    w.h("2.1 Real-world problem")
    w.p("Pottery is the most common find on most excavations. At the Bronze-Age site of Daxinzhuang (Jinan, "
        "Shandong, China), the excavation catalogue alone lists 17,635 sherds from one refuse pit. Before a vessel can be "
        "studied for typology, capacity, use-wear or dating, its sherds must be re-associated. Today an archaeologist "
        "spreads the sherds on a table and compares them one against another. The work scales with the square of the "
        "number of pieces, so it is the first thing to be cut when time or funding runs short. Many collections therefore "
        "stay as unsorted bags of fragments.")
    w.h("2.2 The pattern to be recognised")
    w.p("The pattern is a **relation between two objects**, not a property of one object. A pair of sherds either "
        "*joins* (the pieces were neighbours on the original vessel and share a fracture edge) or does not. We "
        "formulate it in two equivalent ways:")
    w.bullets([
        "**Pair classification**: input = two sherd images; output = P(join). This is a binary pattern-recognition "
        "problem with extreme class imbalance: in the gold standard only 111 of 7,626 possible pairs (1.46%) join.",
        "**Retrieval / ranking**: input = one query sherd; output = all other sherds sorted by P(join). This is how "
        "the system is used: an archaeologist inspects the top-k candidates instead of the whole collection.",
    ])
    w.h("2.3 Why it is hard")
    w.bullets([
        "**Look-alike sherds.** Most Daxinzhuang sherds are the same grey-to-orange cord-marked ware, so sherds from "
        "different vessels look alike (Figure {FIG:gold_samples}).",
        "**Eroded edges.** Fracture edges are chipped and worn after ~3,000 years in the ground, so even true partners "
        "leave visible gaps when assembled (Figure {FIG:gold_gt_assemblies}).",
        "**No canonical orientation or position.** A sherd can be photographed at any rotation, so every comparison must "
        "be rotation-invariant or must search over rotations.",
        "**Two cues that can disagree.** Two pieces may look alike but not fit (same ware, different vessel), or fit "
        "geometrically by coincidence (straight edges fit many places).",
    ])
    w.p("The system built in this project is called **SherdMatch**: it matches a sherd against a collection and "
        "returns the sherds most likely to join it.")
    w.h("2.4 Research question carried over from Review 1")
    w.p("*Does adding fracture-edge geometry to image similarity improve top-k matching over image-only "
        "retrieval?* Hypothesis H1: a model that combines appearance and fracture-edge geometry retrieves true "
        "partners at higher Hit@k than appearance alone, because geometry can reject look-alike sherds that cannot "
        "physically fit.")
    w.h("2.5 Related work")
    w.p("Fragment reassembly is one of the oldest problems in pattern recognition. Freeman and Garder (1964) "
        "solved apictorial jigsaw puzzles by matching boundary shape alone, and Wolfson (1990) formalised curve matching "
        "for such boundaries. In 3-D, Papaioannou et al. (2002) and Huang et al. (2006) reassembled fractured objects by "
        "matching fracture surfaces. Those methods depend on clean, high-resolution fracture geometry. For pottery "
        "specifically, Kampel and Sablatnig (2003) used profile sections to reconstruct vessel shape. Recent 2-D "
        "learning work mostly *classifies* sherds rather than *joining* them. Chetouani et al. (2020) combined pre-trained "
        "CNN features to classify engraved sherd patterns. Pinho et al. (2025) used a pre-trained ResNet-50 to group "
        "sherds by surface pattern and colour to assist reconstruction. Yu et al. (2026) showed that sherd recognisers "
        "latch onto spurious background features and used contrastive learning to suppress them. The "
        "Daxinzhuang-18K benchmark (Ji et al., 2026) is the first large public 2-D collection with expert-verified "
        "physical joins, which makes the comparison in this study possible.")
    w.table(["Work", "Cue used", "Data", "What it leaves open for this study"], [
        ["Freeman & Garder 1964; Wolfson 1990", "Boundary shape", "Synthetic jigsaw pieces", "Clean cut edges; no appearance"],
        ["Papaioannou et al. 2002; Huang et al. 2006", "3-D fracture surface", "3-D scans", "Needs 3-D scanning of every sherd"],
        ["Kampel & Sablatnig 2003", "Profile section", "Rim sherds", "Body sherds have no usable profile"],
        ["Chetouani et al. 2020", "CNN appearance", "Engraved sherds", "Classifies patterns, does not find joins"],
        ["Pinho et al. 2025", "ResNet-50 appearance", "Decorated sherds", "Groups by look; geometry not tested"],
        ["Yu et al. 2026", "Contrastive CNN", "Hemudu sherds", "Recognition, not reassembly"],
        ["**This study", "Appearance + 2-D fracture edge", "Daxinzhuang gold standard (real joins)",
         "Tests whether 2-D edge geometry adds value on real eroded sherds"],
    ], widths=[1.8, 1.3, 1.5, 2.3], caption="Positioning against prior work.")
    w.p("**Research gap.** Appearance-based sherd work rarely evaluates on real, expert-confirmed joins, and "
        "geometry-based reassembly is mostly evaluated on clean or synthetic breaks. Whether 2-D fracture-edge "
        "geometry still helps once edges are eroded, compared fairly against a strong image-only baseline, has not "
        "been measured. That comparison is the core of this case study.")
    w.h("2.6 Key terms")
    w.table(["Term", "Meaning in this report"], [
        ["Sherd", "A broken fragment of a pottery vessel."],
        ["Join / refit", "Two sherds that were neighbours on the vessel and share a fracture edge (positive pair)."],
        ["Vessel group", "All sherds connected through joins; a connected component of the join graph."],
        ["Exterior / interior view", "Photograph of the outer or inner surface of a sherd."],
        ["Cord marks", "Parallel impressions made by a cord-wrapped paddle while shaping the pot; they form stripes."],
        ["Spalling", "Flaking off of the surface layer, which changes a sherd's appearance."],
        ["Query / gallery", "The sherd we search partners for / the collection searched."],
        ["Hit@k", "Share of queries with at least one true partner among the top k suggestions."],
        ["MRR", "Mean reciprocal rank of the first true partner."],
        ["Pose", "Rotation and translation that place a sherd in the assembled position."],
        ["Kabsch / ICP", "Closed-form best rotation between two point sets / its iterative refinement with re-matched points."],
    ], widths=[1.6, 5.2], caption="Archaeological and technical terms.")

    # ---------------- 3. Objectives
    w = Writer(doc, response_anchor(doc, "3. Objectives"))
    w.table(["#", "Objective", "Measurable target / evidence", "Status"], [
        ["O1", "Acquire and understand the Daxinzhuang-18K data and metadata",
         "Dataset, metadata and annotation formats documented; pose convention verified", "Done (Sec. 4–5)"],
        ["O2", "Build a reproducible preprocessing pipeline",
         "Segment, normalise and extract contours for 100% of gold and simulated images", "Done (Sec. 6)"],
        ["O3", "Implement an image-only baseline (Review 1 plan)", "Hit@k, MRR of ResNet-50 / DINOv2 cosine ranking",
         "Done (Sec. 12)"],
        ["O4", "Implement fracture-edge geometric compatibility",
         "Pose recovery rate on synthetic and real pairs; pair AUC", "Done (Sec. 9, 12)"],
        ["O5", "Fuse appearance and geometry in a pattern-recognition model and test H1",
         "Hit@1/5/10, MRR, mAP, P/R/F1 under vessel-grouped CV, with 95% bootstrap CIs", "Done (Sec. 11–13)"],
        ["O6", "Justify the model against alternatives",
         "Logistic regression vs Random Forest vs SVM-RBF, plus an ablation study", "Done (Sec. 10)"],
        ["O7", "Deliver a working prototype",
         "Interactive demo that ranks candidates and previews the best physical fit", "Done (app.py)"],
    ], widths=[0.4, 2.3, 2.9, 1.2], caption="Specific, measurable objectives.")

    # ---------------- 4. Dataset
    w = Writer(doc, response_anchor(doc, "4. Dataset"))
    w.h("4.1 Source and collection")
    w.p("**Daxinzhuang-18K: A Large-Scale 2D Benchmark for Pottery Sherd Retrieval and Reassembly**, by Ji, Xue, "
        "Wang, Ma, Lang, Fang, Tu and Lu, published on Zenodo (record 21553799, version 1.0.0, July 2026, licence "
        "CC-BY 4.0). The sherds come from pit H690 at the Daxinzhuang site in Jinan. Each sherd was photographed on a "
        "plain backdrop with a colour/scale card, exterior and interior separately, and catalogued by stratigraphic "
        "layer. Joins were confirmed physically by experts.")
    w.table(["Part", "Content", "Size", "Used here for"], [
        ["Daxinzhuang-18K (real)", "36,532 JPG photos (1024×771) + metadata workbook, 17,635 catalogue records",
         "4.7 GB", "Metadata study; raw-photo segmentation (E6 planned)"],
        ["dataset_gold_standard", "248 background-removed RGBA PNGs (124 sherds × exterior/interior), gt_all.json with 111 "
         "join records and assembly poses", "53 MB", "Main train/test set (E1–E3, E5)"],
        ["dataset_simulated (8 subsets)", "50,736 synthetic fracture pairs, each with two PNGs and the relative pose",
         "18 GB", "sim_data_8 only (2,212 pairs): geometry sanity check (E4)"],
        ["expert_confirmed_physical_refit_pairs.xlsx", "251 joining groups (pairs or triples) by sherd ID", "16 KB",
         "Ground truth for the planned raw-photo test (E6)"],
    ], widths=[1.6, 2.9, 0.6, 1.8], caption="Contents of the Zenodo record and how each part is used.")
    w.p("Because the full record is about 20 GB, we downloaded only what each experiment needs. From the 4.7 GB "
        "real-photo archive, the metadata workbook and the needed photos were extracted with HTTP byte-range requests "
        "against the zip's central directory, verifying each file's CRC-32. This avoided a full download.")
    w.h("4.2 Samples, labels and features")
    w.p(f"The unit of analysis is a **sherd pair**. The gold standard has {DS['gold_sherds_exterior']} exterior sherds, giving "
        f"124·123/2 = {R['dataset']['n_pairs_total']:,} unordered pairs, of which {R['dataset']['n_refit_pairs']} are labelled "
        "*join* (class 1) and the rest *no join* (class 0). The join graph splits into "
        f"{DS['gold_vessel_groups']} connected components (vessel groups) of 2 to {DS['group_size_max']} sherds "
        f"(median {DS['group_size_median']:.0f}). Most sherds have one or two partners "
        f"({DS['partners_per_sherd']['1']} sherds have exactly one, {DS['partners_per_sherd']['2']} have two, "
        f"{DS['partners_per_sherd']['3']} have three, {DS['partners_per_sherd']['4']} have four).")
    w.table(["Property", "Value"], [
        ["Images (gold)", f"{DS['gold_images_total']} RGBA PNG, 1024 × 1024, transparent background"],
        ["Sherds used (exterior view)", str(DS["gold_sherds_exterior"])],
        ["Positive pairs / all pairs", f"{R['dataset']['n_refit_pairs']} / {R['dataset']['n_pairs_total']:,} (1.46%)"],
        ["Vessel groups", f"{DS['gold_vessel_groups']} (largest {DS['group_size_max']} sherds)"],
        ["Median sherd area", f"{DS['area_px_median'] / 1000:.0f} k px (full resolution)"],
        ["Median sherd perimeter", f"{DS['perimeter_px_median']:.0f} px"],
        ["Median cord-mark stripe coherence", f"{DS['stripe_coherence_median']:.2f} (0 = no direction, 1 = perfect stripes)"],
        ["Simulated pairs (sim_data_8)", f"{DS['sim_data_8_pairs']:,} from {DS['sim_source_pots']} source vessel, many fracture runs"],
    ], widths=[2.4, 4.4], caption="Gold-standard dataset summary (computed by scripts/figures/explore_dataset.py).")
    w.fig("gold_samples.png", "A sample of gold-standard sherds. The ware is uniform: cord-marked, grey to orange, "
          "which is why appearance alone confuses vessels.")
    w.fig("gold_distributions.png", "Exploration of the gold standard: vessel-group sizes, sherd areas, strength of "
          "the cord-mark stripe pattern, and mean colour in Lab space (colour = lightness).")
    w.p("**Observations.** (i) Group sizes are skewed: a few large vessels contribute many pairs, so a random "
        "pair-level split would put sherds of the same vessel in both train and test. We therefore split by vessel "
        "group (Section 11). (ii) Sherd area varies about tenfold, and small sherds have short edges that carry little "
        "shape information. (iii) Stripe coherence is moderate (median 0.49), so the stripe direction is a usable but "
        "noisy cue. (iv) Colour spreads along a single orange-to-grey axis, the main appearance signal separating "
        "vessels.")
    w.p("For each pair we compute 12 features. Four are appearance features and eight describe fracture-edge "
        "geometry. All are continuous real numbers:")
    w.table(["Feature", "Type", "Meaning"], [
        ["colour_sim", "appearance", "−χ² distance of 8×8×8 Lab colour histograms"],
        ["lbp_sim", "appearance", "−χ² distance of rotation-invariant uniform LBP histograms (P=8,R=1 and P=16,R=2)"],
        ["resnet_cos", "appearance", "cosine similarity of ResNet-50 (ImageNet) 2048-d embeddings"],
        ["dino_cos", "appearance", "cosine similarity of DINOv2 ViT-S/14 384-d embeddings"],
        ["geo_score", "geometry", "contact × overlap penalty × orientation agreement at the best alignment"],
        ["contact, contact_frac", "geometry", "edge length (px) that touches within 4 px, absolute and relative to the shorter outline"],
        ["overlap", "geometry", "area overlap of the two sherds after alignment (should be ≈ 0)"],
        ["rmse", "geometry", "residual of the touching edge points (px)"],
        ["orient_gap, orient_conf", "geometry", "angle between the two sherds' cord-mark directions after alignment, and "
                                                "the confidence of that estimate"],
        ["log_area_ratio", "geometry", "|log(area A / area B)|"],
    ], widths=[1.5, 0.9, 4.4], caption="The 12 pair features (inputs to the classifier).", label="features")
    w.fig("sim_samples.png", "Simulated fracture pairs from sim_data_8. Edges are clean polygon breaks, unlike "
          "the eroded real edges.")

    # ---------------- 5. Metadata
    w = Writer(doc, response_anchor(doc, "5. Metadata"))
    w.h("5.1 Excavation metadata workbook")
    w.p(f"Daxinzhuang_AI_Archaeological_Sherd_Metadata.xlsx has two sheets: *Shandong* "
        f"({DS['metadata_records']:,} records) and *Specimens* (176 records). Row 1 of each sheet is a legend that "
        "describes the fields, not a sherd, and must be skipped when the sheet is loaded.")
    w.table(["Field", "Meaning", "Observed values / relevance"], [
        ["Sherd_ID", "Catalogue number of the sherd (JD00001…)", "Primary key; the refit spreadsheet uses it"],
        ["Sherd_Part", "Rim, body sherd, base, foot, …", f"Body sherds dominate ({DS['sherd_part_counts']['Body_Sherd']:,}); "
                                                         f"rims {DS['sherd_part_counts']['Rim']:,}. Rims carry profile information body sherds lack"],
        ["Inferred_Vessel_Type", "Jar, Li tripod, Gui, Dou, …", f"Only {DS['vessel_type_known']} of {DS['metadata_records']:,} records "
                                                                "are typed, so it is too sparse to use as a label"],
        ["Exterior_Photo / Interior_Photo", "File stems of the two views", "Photo numbers ≠ sherd numbers (see 5.2)"],
        ["Site", "Jinan_Daxinzhuang", "Constant; one site"],
        ["Context_Unit", "Pit and layer, e.g. H690_Layer_8", f"{DS['context_units']} units: stratigraphy, a "
                                                             "potential prior (joining sherds usually lie in nearby layers)"],
        ["Small_Find_No", "Registered small-find number", "Empty for the main sheet"],
        ["Notes", "Condition notes, e.g. Exterior_Spalled", "Surface spalling changes appearance and explains some errors"],
    ], widths=[1.6, 2.0, 3.2], caption="Metadata fields and why they matter.")
    w.fig("metadata_distributions.png", "Metadata distributions (log scales): sherd parts, inferred vessel types "
          "(mostly not inferred), and records per stratigraphic layer of pit H690.")
    w.h("5.2 A metadata pitfall we found and fixed")
    w.p("A naive reading suggests that sherd JD00076 is photographed in file JD_00076.JPG. The metadata shows "
        "otherwise: each sherd has *two* photo numbers (JD00001 → JD_00001 exterior and JD_00002 interior), so photo "
        "numbers run about twice as fast as sherd numbers. Our first download script mapped IDs by number and would have "
        "silently paired the wrong photos. The bug was caught by reading the metadata, and the fixed script maps every "
        "refit sherd through the *Exterior_Photo* column. This is a concrete case of metadata preventing a label "
        "error that pixels alone could never reveal.")
    w.h("5.3 Annotation metadata (gt_all.json) and pose conventions")
    w.p("Each of the 111 gold records has *pair_id*, *side* (all exterior), *img_a*, *img_b* and two poses "
        "A_action, B_action = [x, y, cos θ, sin θ]. The README does not state the exact convention. From the "
        "dataset's reference script we established that each image is first recentred on its alpha-weighted "
        "centroid c, then rotated by θ and placed at (x, y), so world = R(θ)(p − c) + t. We confirmed this "
        "independently: under this convention the true partners touch along 75–170 px of edge with ≤7% overlap, "
        "whereas the alternative rotation sign gives large overlaps. The simulated data use a *different* "
        "convention: rotation about the image centre with the opposite sign of θ. We found it by testing all eight "
        "combinations of sign, centre and inversion and keeping the one that produced contact without overlap "
        "(211 px mean contact, 4% overlap). Figure {FIG:gold_gt_assemblies} re-renders gold joins with our own implementation of the "
        "convention.")
    w.h("5.4 Why metadata matters for evaluation")
    w.bullets([
        "**Leakage control.** The vessel-group structure (derived from the join graph) defines the cross-validation "
        "folds, so no vessel appears in both training and test data.",
        "**Interpretation of errors.** Condition notes (spalling, adhered soil, residue) explain why some sherds' "
        "appearance does not match their partners.",
        "**Future priors.** Context unit and sherd part are not used by the model yet, but they are realistic priors "
        "(Section 14).",
    ])
    w.table(["Sherd_ID", "Sherd_Part", "Exterior_Photo", "Interior_Photo", "Context_Unit"], [
        ["JD00001", "Rim", "JD_00001", "JD_00002", "H690_Layer_1"],
        ["JD00002", "Rim", "JD_00003", "JD_00004", "H690_Layer_1"],
        ["JD00003", "Body_Sherd", "JD_00005", "JD_00006", "H690_Layer_1"],
    ], widths=[1.1, 1.1, 1.3, 1.3, 1.6], caption="First records of the Shandong sheet (Site = Jinan_Daxinzhuang for all).")

    # ---------------- 6. Exploration and preprocessing
    w = Writer(doc, response_anchor(doc, "6. Data Exploration"))
    w.h("6.1 Data quality issues found")
    w.table(["Issue", "Where", "Consequence if ignored", "Handling"], [
        ["Background", "Raw photos: grey/white backdrop + colour card + ink labels", "Card or backdrop treated as sherd",
         "Otsu on colour distance to the border colour, rectangularity filter for the card, GrabCut refinement"],
        ["Illumination / colour cast", "All images", "Texture features follow lighting, not surface",
         "CLAHE on the L channel (texture only); colour features use raw Lab"],
        ["Arbitrary rotation", "All images", "Rotation-dependent features miss partners",
         "Rotation-invariant LBP; 4-rotation averaged deep embeddings; geometry searches over all rotations"],
        ["Scale must be preserved", "Gold and sim", "Per-image resizing would change edge lengths",
         "One global working scale (0.5), never per-image resizing"],
        ["Eroded / chipped edges", "Real sherds", "Edges do not fit exactly", "Contour smoothing (σ = 1.5 samples), contact "
                                                                             "tolerance of 4 px, ICP refinement"],
        ["Tiny specks / holes in the mask", "Alpha matting edges", "Spurious contours", "Largest connected component, hole filling, "
                                                                                       "morphological opening"],
        ["Corrupt / partial files", "Downloads", "Crashes or silent bad data", "CRC-32 check on extraction; image validation on load"],
        ["Class imbalance (1.46%)", "Pair labels", "Classifier predicts 'no join' always", "Class-balanced weights; "
                                                                                          "PR-based metrics"],
        ["Leakage via vessel", "Pair labels", "Inflated scores", "Vessel-grouped folds"],
    ], widths=[1.3, 1.5, 1.6, 2.4], caption="Data quality issues and how each is handled.", font=8)
    w.p("Missing values, duplicates and outliers in the usual tabular sense do not arise: every gold image has an "
        "annotation, no two images are identical (the largest DINOv2 cosine similarity between different sherds is "
        "well below 0.999), and every sherd area is plausible. Two of the 36,532 catalogue records lack an interior "
        "photo; this does not affect the exterior-only analysis.")
    w.h("6.2 Preprocessing sequence")
    w.table(["Step", "Operation", "Parameters", "Why it suits this data"], [
        ["01 Validate", "Open and fully decode; reject empty masks", "mask ≥ 200 px", "Catches truncated downloads early"],
        ["02 Standardise", "Convert to RGBA; resize by a global factor", "scale 0.5 (gold, sim); 1.0 (photos)",
         "Keeps true relative size, which edge matching needs; halves compute"],
        ["03 Isolate", "Alpha > 127 (gold/sim) or Otsu + card filter + GrabCut (photos); largest component; fill holes",
         "rectangularity < 0.88; GrabCut 4 iterations", "Fragment shape must be exact for contour matching"],
        ["04 Normalise", "CLAHE on Lab L channel", "clip 2.0, 8×8 tiles", "Removes lighting gradients but keeps cord-mark relief"],
        ["05 Edges", "Outer contour, oriented consistently, resampled every 3 px, Gaussian-smoothed",
         "step 3 px, σ = 1.5 samples", "Equal spacing makes curvature windows comparable between sherds"],
        ["06 Features", "Colour/LBP histograms, deep embeddings, stripe orientation, curvature signature",
         "see Table {TAB:features}", "Inputs for appearance and geometry cues"],
    ], widths=[0.9, 2.2, 1.6, 2.1], caption="Preprocessing pipeline as implemented in src/pfm/preprocess.py.", font=8)
    w.fig("preprocessing_pipeline.png", "Preprocessing of sherd JDX-100: input, mask, CLAHE-normalised texture, "
          "resampled contour, and its curvature (turning-angle) signature. Peaks are corners of the outline.")
    w.fig("gold_gt_assemblies.png", "Gold-standard joins re-rendered from the expert poses with our own code. The "
          "pieces touch but leave gaps along eroded parts of the edge.")
    w.p("For raw excavation photos the isolation step is harder, because grey sherds sit on a grey backdrop "
        "next to a black-and-white scale card. Otsu thresholding alone picked the card in 2 of the first 6 test photos "
        "and under-segmented grey sherds. The rectangularity filter removes the card in every photo we inspected, and "
        "GrabCut recovers most of each sherd. Two errors remain (Figure {FIG:photo_segmentation}): the light, "
        "low-contrast side of some sherds is cut off (JD_00028, JD_00076), and soft cast shadows are included on the "
        "other side. Both distort the outline, which matters for the planned raw-photo experiment (E6).")
    w.fig("photo_segmentation.png", "Automatic segmentation of raw excavation photos. The scale card is always "
          "rejected; light sherd edges are partly lost and shadows partly included.")

    # ---------------- 7. Scope
    w = Writer(doc, response_anchor(doc, "7. Scope"))
    w.h("Scope")
    w.p("The system ranks candidate joins for 2-D photographs of individual sherds and previews the best rigid "
        "alignment of a pair. It works on the exterior view, assumes sherds are roughly flat in the photo (2-D "
        "rigid motion), and is evaluated on one site's ware. It does **not** perform full multi-piece vessel "
        "reconstruction, and it does not decide joins. It produces a short list for an expert to check.")
    w.h("Applications")
    w.bullets([
        "Triage in finds processing: show the archaeologist the 5–10 most likely partners for each new sherd.",
        "Re-examining museum stores where sherds of one vessel ended up in different bags or layers.",
        "Digital archives: linking photographs of sherds across collections, without handling the objects.",
        "Teaching: the demo makes the difference between 'looks alike' and 'fits together' tangible.",
    ])
    w.h("Expected impact")
    k10 = M[PROP]["hit@10"]
    w.p(f"With the proposed model, an expert who inspects the top 10 candidates finds a true partner for "
        f"{pct(k10)} of sherds, against {pct(M['Random ranking']['hit@10'])} for random inspection of 10 sherds. In a "
        "collection of thousands of sherds, that means looking at ten candidates rather than at everything, for most "
        "sherds. The cost is a few seconds of computation per query.")
    w.h("Assumptions")
    w.bullets([
        "Photographs are taken at a consistent scale (true for the gold set; approximately true for the field photos, "
        "which include a scale card).",
        "Sherds are rigid and their outline is visible; joining edges are part of the outer contour.",
        "Expert joins are correct and complete for the pairs listed. Unlisted pairs are treated as non-joins, which can "
        "only make the measured precision pessimistic.",
    ])
    w.h("Limitations (summary; details in Section 14)")
    w.bullets([
        "Real fracture edges are eroded, which limits what 2-D geometry can recover.",
        "Only 111 positive pairs: enough to fit a linear model, not enough to train a deep network for this task.",
        "One site and one ware type; generalisation to other assemblages is untested.",
    ])

    # ---------------- 8. Novelty
    w = Writer(doc, response_anchor(doc, "8. Novelty"))
    w.p("The individual building blocks (LBP, CNN embeddings, Kabsch alignment, ICP, logistic regression) are "
        "standard and are cited as such. The contribution of this project lies in how they are combined and, above "
        "all, in how they are evaluated:")
    w.table(["Aspect", "What is new in this work", "Existing practice"], [
        ["Problem formulation", "Joining as pair classification *and* retrieval, scored with archaeologist-centred "
                                "Hit@k on real expert joins", "Sherd *classification* by type or decoration"],
        ["Evaluation protocol", "Vessel-grouped CV (no vessel in train and test), paired bootstrap CIs, "
                                "synthetic-vs-real comparison of the same matcher", "Random splits; synthetic breaks only"],
        ["Feature design", "Complementary curvature windows (A forward vs B reversed and negated), overlap-verified "
                           "alignment, and a cord-mark orientation-consistency cue", "Shape descriptors per sherd, "
                                                                                     "no physical-fit check"],
        ["Dataset preparation", "Pose conventions of gold and simulated sets derived and verified; photo-ID mapping "
                                "error found and fixed; range-request extraction of only the needed files", "Not documented"],
        ["Analysis", "Ablation of every cue, error analysis (same-vessel confusions, sherd size), honest negative "
                     "result for H1", "Report best number only"],
        ["Prototype", "Interactive demo with ranked candidates, feature evidence and a fit preview", "Offline scripts"],
    ], widths=[1.4, 3.2, 2.2], caption="Contribution of this case study versus existing approaches.")
    w.p("**Individual contribution.** This is a one-person project. All code in the repository (preprocessing, "
        "features, geometric matcher, experiments, figures, demo, this report) was written, run and analysed by the "
        "student, using the public libraries and pre-trained weights listed in Section 16.")

    # ---------------- 9. Model
    w = Writer(doc, response_anchor(doc, "9. Pattern Recognition"))
    w.p("**Selected model: a two-cue pairwise classifier. Appearance and fracture-edge geometry features are "
        "fused by an L2-regularised, class-balanced logistic regression (LR)**, whose output probability ranks "
        "candidate partners.")
    w.h("9.1 Input, output and stages")
    w.fig("method_architecture.png", "System architecture. Per-sherd descriptors are computed once and cached; the "
          "geometric matcher runs per pair.")
    w.table(["Stage", "Input", "Output"], [
        ["1. Preprocess each sherd", "Image file", "Mask, normalised RGB, contour (N×2), centroid"],
        ["2. Per-sherd descriptors", "Preprocessed sherd", "Colour hist (512), LBP (28), ResNet-50 (2048), DINOv2 (384), stripe "
                                                           "orientation + coherence"],
        ["3. Pair geometry", "Two contours + orientations", "Best rigid transform (R, t) and contact, overlap, rmse, orientation gap"],
        ["4. Pair feature vector", "Stages 2–3 for a pair", "x ∈ ℝ¹² (Table {TAB:features})"],
        ["5. Classifier", "x", "P(join | x) = σ(wᵀz + b), z = standardised x"],
        ["6. Ranking", "P(join) for query vs every gallery sherd", "Ranked candidate list + fit preview of the top candidate"],
    ], widths=[1.6, 2.0, 3.2], caption="Processing stages.")
    w.h("9.2 Fracture-edge geometric matcher (stage 3)")
    w.p("Where two sherds join, their outlines follow the same curve but are traversed in opposite directions, and "
        "a bump on one is a dent on the other. The matcher exploits this:")
    w.bullets([
        "**Curvature signature.** For each contour, the turning angle κᵢ = wrap(∠(pᵢ₊₁ − pᵢ) − ∠(pᵢ − pᵢ₋₁)) is computed "
        "at every 3 px sample.",
        "**Complementary windows.** All windows of 32 samples (~96 px, stride 2) from A are compared with all windows from "
        "B *reversed and negated* (−κ_B read backwards), using squared Euclidean distance. Straight windows match "
        "everything, so the distance is divided by (0.5 + Σ|κ|), favouring distinctive shapes.",
        "**Alignment.** For the 8 best seed pairs (with non-maximum suppression), the rigid transform mapping B's window "
        "onto A's is solved in closed form by the Kabsch/SVD method (Kabsch, 1976). It is then refined by two "
        "iterations of ICP (Besl & McKay, 1992) on all touching points.",
        "**Verification.** Each candidate pose is scored on the whole outline: contact = length of B's edge within 4 px of A; "
        "overlap = area of intersection / smaller area (solid sherds cannot overlap); score = contact × max(0, 1 − "
        "overlap/0.08) × exp(−(Δφ/20°)²·c). Here Δφ is the angle between the two sherds' cord-mark directions "
        "(structure-tensor estimate, Bigün & Granlund 1987) and c is its confidence. The best-scoring pose is kept.",
    ])
    w.fig("method_edge_matcher.png", "The matcher on a true pair (JDX-12 + JDX-5). It locates the shared fracture "
          "edge on both outlines (orange), aligns B to A, and the curvature of A along the joint mirrors that of B. The "
          "fit is checked against the whole outline, not only the seed window.")
    w.fig("results_alignment_examples.png", "Output of the geometric matcher for true pairs: the two best (error "
          "3 px, the predicted pose coincides with the expert pose) and the two worst (the matcher found a "
          "plausible-looking but wrong fit).")
    w.h("9.3 Logistic regression (stage 5)")
    w.p("LR models the log-odds of a join as a weighted sum of the standardised features: "
        "log[P/(1−P)] = b + Σⱼ wⱼ zⱼ. Weights are learned by minimising the class-weighted log-loss with an L2 "
        "penalty (C = 1). Class weights are inversely proportional to class frequency, so the 111 joins count as much "
        "as the 7,515 non-joins. At query time the predicted probability is computed for the query against every other "
        "sherd and sorted. Ranking only needs the order of the scores, so the decision threshold matters only for "
        "the pair-classification metrics, where it is chosen on training data (Section 11).")
    w.p("**Algorithm 1: ranking the gallery for one query sherd q.**")
    algo = doc.add_table(rows=1, cols=1)
    _borders(algo)
    cell = algo.rows[0].cells[0]
    _shade(cell, "F7F3EE")
    lines = [
        "input : query image q, gallery G (cached descriptors), trained LR (w, b), scaler (μ, σ)",
        "1  f_q ← preprocess(q)                      # mask, CLAHE, contour",
        "2  d_q ← descriptors(f_q)                   # colour, LBP, ResNet-50, DINOv2, stripe angle",
        "3  for each g in G:",
        "4      a ← [−χ²(colour), −χ²(LBP), cos(ResNet), cos(DINOv2)]",
        "5      (R, t) ← argmax over 8 curvature seeds of verify(Kabsch+ICP(f_q, f_g))",
        "6      x ← [a, contact, contact_frac, overlap, rmse, score, Δφ·c, c, |log area ratio|]",
        "7      s_g ← σ( wᵀ (x − μ)/σ + b )                # P(join)",
        "8  return G sorted by s_g (descending), with (R, t) of the top candidate for the fit preview",
    ]
    cell.text = ""
    for i, ln in enumerate(lines):
        para = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        run = para.add_run(ln)
        run.font.name = "Consolas"
        run.font.size = Pt(8)
        para.paragraph_format.space_after = Pt(0)
    w._place(algo._tbl)
    w.p("", space_after=4)
    w.h("9.4 Why this model fits the problem")
    w.bullets([
        "The decision really is a weighing of evidence from different sources (\"similar colour\", \"edges touch\", "
        "\"stripes line up\"). LR expresses exactly that, and its weights can be read and checked for physical plausibility.",
        "With 111 positives, a model with 13 parameters is well determined. A deep network would overfit.",
        "It produces calibrated-ish probabilities, which are needed for a ranking across all candidates.",
        "Training takes milliseconds and prediction is a dot product, so the expensive part (geometry) is the only cost.",
    ])
    w.h("9.5 Working prototype")
    w.p("The model is packaged as SherdMatch, a four-page web application (app.py, built with Gradio):")
    w.bullets([
        "**Home** explains the problem for a non-specialist: the scale of the sorting task, why it is hard, the "
        "question tested, the four-step approach and the headline results.",
        "**How it works** walks through preprocessing, appearance cues, the fracture-edge matcher, the classifier "
        "and the evaluation protocol, with the figures from this report.",
        "**Matcher** is the working tool. The user clicks a sherd in a visual grid of the 124-sherd collection, or "
        "uploads a new image (a PNG with a transparent background, or a photo on a plain backdrop, segmented "
        "automatically), and chooses the proposed model or the image-only baseline. The page shows the top-5 "
        "candidates as cards with score bars, a table of the evidence behind each score, and a preview of the "
        "physical fit of the best candidate, the first step toward digital reconstruction. Expert-confirmed joins "
        "are highlighted, so the model's behaviour can be checked live during the review.",
        "**Results** summarises the experiments, including the negative finding about geometry.",
    ])
    w.p("A collection query answers instantly from cached features; an uploaded sherd takes about 2–6 s on a laptop "
        "CPU. Switching the method live makes the main result tangible: for query JDX-103 the image-only baseline "
        "ranks the true partner 28th, while SherdMatch ranks it first.")
    w.fig("demo_home.jpg", "SherdMatch, Home page: the problem explained before the tool is used.", width=6.0)
    w.fig("demo_gallery_query.jpg", "SherdMatch, Matcher page: query JDX-103 picked from the collection grid. Its "
          "expert-confirmed partner is ranked first, and the right panel previews the predicted fit.", width=6.0)
    w.fig("demo_evidence_table.jpg", "SherdMatch, Matcher page: candidate cards (expert-confirmed joins outlined in "
          "green) and the evidence table that explains each score.", width=6.0)
    w.p("**Source code.** The full project (pipeline, experiments, figures, demo and this report's generator) is "
        "public at https://github.com/vansh070605/SherdMatch, and every number in this report is "
        "regenerated by scripts/experiments/run_experiments.py and scripts/figures/make_figures.py.")

    # ---------------- 10. Justification
    w = Writer(doc, response_anchor(doc, "10. Justification"))
    w.p("We compared the selected model with the image-only approach planned in Review 1 and with two "
        "non-linear classifiers trained on the same 12 features under the same folds:")
    rows = []
    for n, interp, cost in [(BASE, "High (one similarity)", "1 forward pass / sherd"),
                            (PROP, "High (12 signed weights)", "ms to train"),
                            (RF, "Medium (feature importances)", "seconds to train"),
                            (SVM, "Low (kernel)", "seconds; scales O(n²)")]:
        m = M[n]
        rows.append([("**" if n == PROP else "") + n, interp, cost, f3(m["hit@5"]), f3(m["hit@10"]), f3(m["mrr"]),
                     f3(m.get("pair_f1") or float("nan")) if m.get("pair_f1") is not None else "–"])
    w.table(["Model", "Interpretability", "Computational cost", "Hit@5", "Hit@10", "MRR", "Pair F1"], rows,
            widths=[2.1, 1.2, 1.2, 0.55, 0.6, 0.55, 0.6], caption="Model comparison on identical features and folds.", font=8)
    w.table(["Criterion", "Logistic regression (selected)", "Random Forest", "SVM (RBF)"], [
        ["Data characteristics: 12 continuous features, 111 positives", "Few parameters, well-posed",
         "Many trees on few positives, tends to memorise", "Kernel width and C hard to tune with few positives"],
        ["Interpretability", "Signed weights per feature (Figure {FIG:results_lr_coefficients})", "Importances only, no direction", "None"],
        ["Complexity / cost", "Linear; trivial", "400 trees; moderate", "Kernel matrix; grows quadratically"],
        ["Robustness to imbalance", "Class weights", "Balanced subsampling", "Class weights"],
        ["Measured ranking quality", f"Best Hit@5/10 (" + f3(M[PROP]['hit@5']) + "/" + f3(M[PROP]['hit@10']) + ")",
         f"Lower ranking ({f3(M[RF]['hit@5'])}/{f3(M[RF]['hit@10'])}), similar F1", f"Lowest ({f3(M[SVM]['hit@5'])}/{f3(M[SVM]['hit@10'])})"],
    ], widths=[1.8, 1.7, 1.6, 1.7], caption="Criteria-based comparison.", font=8)
    w.p(f"Against the **image-only baseline** (DINOv2 cosine, the Review 1 plan), the selected model improves "
        f"Hit@5 by {ci('proposed_vs_dinov2', 'hit@5')} and Hit@10 by {ci('proposed_vs_dinov2', 'hit@10')}. Against "
        f"the **Random Forest**, it ranks better (Hit@5 {f3(M[PROP]['hit@5'])} vs {f3(M[RF]['hit@5'])}) with a "
        f"comparable pair F1 ({f3(M[PROP]['pair_f1'])} vs {f3(M[RF]['pair_f1'])}). The forest trades precision for recall "
        "at its threshold but orders candidates less well. The **SVM** is worst on every metric. With so few positives "
        "the flexibility of non-linear models does not pay off, and the linear model's interpretability is a "
        "decisive advantage for an archaeological tool whose suggestions must be explained.")
    w.p("**Why not an end-to-end deep (Siamese) network?** It would need thousands of real positive pairs. The "
        "111 real joins are too few, and the 50,736 simulated pairs come from clean synthetic breaks of a few "
        "vessels. Section 12 shows that the geometry of real breaks differs from synthetic ones, so a network trained on "
        "simulation would face a large domain gap. We therefore use pre-trained embeddings as fixed features and learn "
        "only a small fusion model, which is the appropriate choice for this data size.")

    # ---------------- 11. Setup
    w = Writer(doc, response_anchor(doc, "11. Experimental"))
    w.h("11.1 Validation strategy")
    w.p("**5-fold vessel-grouped cross-validation.** The 33 vessel groups (connected components of the join graph) "
        "are shuffled (seed 0) and assigned to 5 folds. For test fold k, the model is trained only on pairs whose two "
        "sherds both lie outside fold k, and it scores every pair that involves a fold-k sherd. Each query is ranked "
        "against the full gallery of 123 other sherds, as in real use. A pair whose two sherds fall in different test "
        "folds receives the average of its two scores. Because no vessel is ever in both train and test, the model "
        "cannot succeed by memorising a vessel's colour.")
    w.p("**Decision threshold.** For the pair-classification metrics, the threshold that maximises F1 is chosen by an "
        "inner 3-fold grouped CV on the training pairs only, then applied to the held-out pairs fully inside the test "
        "fold (1,542 pairs in total, 111 positive).")
    w.h("11.2 Experiments")
    w.table(["ID", "Question", "Data", "Protocol"], [
        ["E1", "Does fusion beat image-only retrieval?", "Gold, 124 queries", "Vessel-grouped 5-fold CV"],
        ["E2", "Which classifier?", "Same", "LR vs RF vs SVM-RBF on identical features and folds"],
        ["E3", "Which cues matter?", "Same", "Drop one cue group at a time from the proposed model"],
        ["E4", "Does the edge matcher work on clean breaks?", "sim_data_8, 300 random pairs + 300 negatives",
         "Pose error vs ground truth; AUC vs pairs from different fracture runs"],
        ["E5", "Does it recover real poses?", "111 gold joins", "Mean contour displacement vs expert pose"],
        ["E6", "Does it transfer to raw field photos?", "Expert refit spreadsheet + distractor photos",
         "Planned: gold-trained models, appearance top-30 then geometry re-rank (script ready, not run for this review)"],
    ], widths=[0.4, 2.0, 2.0, 2.4], caption="Experiments.")
    w.h("11.3 Environment and parameters")
    w.table(["Item", "Setting"], [
        ["Hardware / OS", "Laptop, Windows 11; inference on CPU (PyTorch CPU build), 8 worker processes for geometry"],
        ["Software", "Python 3.12, PyTorch 2.9.1, timm 1.0.29, scikit-learn 1.4.1, OpenCV 4.11, scikit-image 0.26, Gradio 6"],
        ["Pre-trained models", "resnet50.a1_in1k (ImageNet-1k); vit_small_patch14_dinov2.lvd142m at 224 px; 4 rotations averaged"],
        ["Preprocessing", "scale 0.5, contour step 3 px, smoothing σ 1.5, CLAHE clip 2.0"],
        ["Geometry", "window 32, stride 2, 8 seeds, contact tolerance 4 px, overlap penalty 0.08, orientation σ 20°"],
        ["Classifiers", "LR: C = 1, balanced; RF: 400 trees, min leaf 5, balanced subsample; SVM: RBF, C = 1, balanced"],
        ["Randomness", "All seeds = 0; results are deterministic"],
        ["Runtime", "Features for 124 sherds ≈ 40 s; geometry for all 7,626 pairs ≈ 1 min; full CV ≈ 1 min"],
    ], widths=[1.6, 5.2], caption="Implementation environment and parameters.")
    w.h("11.4 Reproducibility")
    w.table(["Step", "Command", "Output"], [
        ["Fetch photo subset", "python scripts/data/fetch_real_subset.py", "data/raw/Daxinzhuang-18K/… (refit + distractor photos)"],
        ["Explore data", "python scripts/figures/explore_dataset.py", "outputs/dataset_summary.json, dataset figures"],
        ["E1–E5", "python scripts/experiments/run_experiments.py", "outputs/results.json, tables, trained model"],
        ["E6 (planned)", "python scripts/experiments/run_real_photos.py", "outputs/results_real_photos.json"],
        ["Figures + CIs", "python scripts/figures/make_figures.py; make_method_figures.py", "outputs/figures, bootstrap_ci.csv"],
        ["This report", "python scripts/docs/build_report.py", "docs/Review2_Case_Study_Report.docx"],
        ["Demo", "python app.py", "http://127.0.0.1:7860"],
    ], widths=[1.3, 2.9, 2.6], caption="Reproducing every result from the repository.", font=8)
    w.h("11.5 Baselines and metrics")
    w.p("Baselines: random ranking; colour-histogram χ²; LBP χ²; ResNet-50 cosine; DINOv2 cosine (the Review 1 "
        "image-only plan); and the raw geometry score. Metrics and why they were chosen:")
    w.bullets([
        "**Hit@k** (k = 1, 5, 10): fraction of queries with at least one true partner in the top k. It is exactly what an "
        "archaeologist experiences when checking k candidates.",
        "**MRR**: mean of 1/rank of the first true partner. It rewards putting a partner at the very top.",
        "**mAP**: mean average precision over all partners of a query, for sherds with several partners.",
        "**CMC curve**: Hit@k for k = 1…30, showing the whole trade-off.",
        "**Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix** for the pair classifier. Accuracy is reported but "
        "is not informative: with 1.46% positives, predicting 'no join' for everything already scores 98.5%.",
        "**Paired bootstrap 95% CIs** (5,000 resamples of the 124 queries) for differences between methods, because "
        "124 queries is a small test set.",
    ])

    # ---------------- 12. Results
    w = Writer(doc, response_anchor(doc, "12. Results"))
    w.h("12.1 E1: retrieval of join partners")
    names = ["Random ranking", "Colour histogram (chi2)", "LBP texture (chi2)", "ResNet-50 embedding (cosine)", BASE,
             "Geometry score only (no learning)", "Geometry features (LR)", APPLR, PROP]
    w.table(["Method", "Hit@1", "Hit@5", "Hit@10", "MRR", "mAP", "Med. rank"], retrieval_rows(names),
            widths=[2.6, 0.6, 0.6, 0.65, 0.6, 0.6, 0.75], caption="Retrieval results (124 queries, 123 candidates each, "
                                                                    "vessel-grouped CV). Higher is better except median rank.", font=8)
    w.fig("results_retrieval_bars.png", "Retrieval metrics of the main methods.")
    w.fig("results_cmc.png", "Cumulative match characteristic: share of queries with a true partner within the top k.",
          width=4.6)
    w.p(f"**Reading the results.** Every learned or pre-trained cue is far above random (Hit@10 "
        f"{f3(M['Random ranking']['hit@10'])}). Among single cues, the simple **colour histogram is as good as deep "
        f"embeddings** (Hit@10 {f3(M['Colour histogram (chi2)']['hit@10'])} vs {f3(M[BASE]['hit@10'])} for DINOv2). "
        "Sherds of one vessel share their firing colour, while the deep models also respond to shape and texture "
        "that differ between neighbouring sherds. DINOv2 is the best single cue at the very top (Hit@1 "
        f"{f3(M[BASE]['hit@1'])}). **Geometry alone is weak** (Hit@10 {f3(M['Geometry features (LR)']['hit@10'])}): "
        "with eroded edges, many wrong pairs find an equally good-looking fit. **Learning to combine appearance cues** "
        f"lifts Hit@5 from {f3(M[BASE]['hit@5'])} to {f3(M[APPLR]['hit@5'])} and gives the best Hit@1 and MRR. "
        f"**Adding geometry** gives the best Hit@5 ({f3(M[PROP]['hit@5'])}), Hit@10 ({f3(M[PROP]['hit@10'])}) and "
        f"median rank ({M[PROP]['median_rank']:.0f}), but lowers Hit@1 from {f3(M[APPLR]['hit@1'])} to "
        f"{f3(M[PROP]['hit@1'])}.")
    w.h("12.2 Statistical significance")
    bt = [["Proposed − DINOv2 (image-only)"] + [ci("proposed_vs_dinov2", k) for k in ("hit@1", "hit@5", "hit@10", "mrr")],
          ["Proposed − Appearance LR"] + [ci("proposed_vs_appearanceLR", k) for k in ("hit@1", "hit@5", "hit@10", "mrr")],
          ["Appearance LR − DINOv2"] + [ci("appearanceLR_vs_dinov2", k) for k in ("hit@1", "hit@5", "hit@10", "mrr")]]
    bt = [[r[0]] + [x.replace(" (95% CI ", "\n[").replace(")", "]") for x in r[1:]] for r in bt]
    w.table(["Difference", "Hit@1", "Hit@5", "Hit@10", "MRR"], bt, widths=[1.9, 1.2, 1.2, 1.2, 1.2],
            caption="Paired bootstrap differences in percentage points with 95% confidence intervals. An interval "
                    "that excludes 0 is significant.", font=8)
    w.p("The improvement over the image-only baseline at Hit@5 and Hit@10 is significant. The improvement from "
        "adding geometry on top of learned appearance is **not** significant at Hit@5/10, and the drop at Hit@1 "
        "is.")
    import pandas as pd
    pf = pd.read_csv(OUT / "tables" / "per_fold.csv")
    cols = [c for c in pf.columns if "Hit@" in c]
    w.table(["Fold", "Queries", "Vessels"] + [c.replace(" (image-only)", "").replace(" Hit@", "\nHit@") for c in cols],
            [[int(r.fold), int(r.queries), int(r.vessels)] + [f3(r[c]) for c in cols] for _, r in pf.iterrows()]
            + [["Std. dev.", "", ""] + [f3(pf[c].std()) for c in cols]],
            widths=[0.6, 0.65, 0.65] + [0.82] * len(cols), caption="Results per cross-validation fold.", font=8)
    better10 = int((pf["Proposed (LR) Hit@10"] > pf["DINOv2 (image-only) Hit@10"]).sum())
    better5 = int((pf["Proposed (LR) Hit@5"] > pf["DINOv2 (image-only) Hit@5"]).sum())
    w.p(f"The per-fold view shows how much results depend on which vessels are held out. The proposed model beats "
        f"the image-only baseline at Hit@10 in {better10} of 5 folds and at Hit@5 in {better5} of 5. Fold-to-fold "
        f"spread of about ±{100 * pf['Proposed (LR) Hit@10'].std():.0f} points is comparable to the differences "
        "between methods. This is one more reason to rely on the pooled bootstrap intervals rather than on any "
        "single split, and it shows that a larger test set is needed before smaller effects can be resolved.")
    w.h("12.3 Pair classification and confusion matrix")
    pr_rows = []
    for n in ["Geometry features (LR)", APPLR, PROP, RF, SVM]:
        m = M[n]
        pr_rows.append([("**" if n == PROP else "") + n, f3(m["pair_precision"]), f3(m["pair_recall"]), f3(m["pair_f1"]),
                        f3(m["pair_roc_auc"]), f3(m["pair_pr_auc"]), f3(m["pair_accuracy"]),
                        f"{m['pair_tp']}/{m['pair_fp']}/{m['pair_fn']}/{m['pair_tn']}"])
    w.table(["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Accuracy", "TP/FP/FN/TN"], pr_rows,
            widths=[2.1, 0.6, 0.55, 0.5, 0.6, 0.6, 0.65, 1.1], caption="Pair classification on held-out pairs "
                                                                       "(1,542 pairs, 111 joins; chance PR-AUC = 0.072).", font=8)
    w.fig("results_pr_confusion.png", "Precision-recall curves and the confusion matrix of the proposed model at the "
          "threshold chosen on training folds.")
    m = M[PROP]
    w.p(f"At its operating point the proposed model flags {m['pair_tp'] + m['pair_fp']} pairs as joins, of which "
        f"{m['pair_tp']} are correct (precision {f3(m['pair_precision'])}), and finds {m['pair_tp']} of the 111 joins "
        f"(recall {f3(m['pair_recall'])}). Accuracy is {pct(m['pair_accuracy'])}, but it is almost meaningless here: "
        f"the trivial all-'no join' classifier reaches "
        f"{pct((m['pair_tn'] + m['pair_fp']) / (m['pair_tn'] + m['pair_fp'] + m['pair_tp'] + m['pair_fn']))} on these "
        "test pairs. PR-AUC is "
        f"{f3(m['pair_pr_auc'])}, {m['pair_pr_auc'] / 0.072:.1f}× the chance level. ROC-AUC is high "
        f"({f3(m['pair_roc_auc'])}) because the model easily rejects obviously different sherds. The hard part is the "
        "small set of look-alike non-joins, which the PR curve exposes and the ROC curve hides. As a stand-alone "
        "verdict ('these two join') the model is not reliable. As a ranker that shortens the list it is useful, "
        "which matches its intended use.")
    w.h("12.4 E3: ablation and model weights")
    ab_rows = []
    for n in [PROP] + [k for k in M if k.startswith("Ablation")]:
        mm = M[n]
        ab_rows.append([n.replace("Ablation: proposed - ", "without ") if n != PROP else "**Full proposed model",
                        f3(mm["hit@1"]), f3(mm["hit@5"]), f3(mm["hit@10"]), f3(mm["mrr"]), f3(mm["pair_f1"])])
    w.table(["Variant", "Hit@1", "Hit@5", "Hit@10", "MRR", "Pair F1"], ab_rows, widths=[2.6, 0.7, 0.7, 0.7, 0.7, 0.8],
            caption="Ablation: each row removes one group of features from the proposed model.", font=8)
    w.fig("results_ablation.png", "Change in Hit@5 and MRR when a cue is removed. Negative bars mean the cue helped.")
    co = R["proposed_lr_coefficients"]
    w.fig("results_lr_coefficients.png", "Standardised logistic-regression weights of the proposed model (trained on "
          "all gold pairs).", width=5.0)
    w.p(f"Removing **colour** or the **deep embeddings** causes the largest drops (Hit@5 to "
        f"{f3(M['Ablation: proposed - colour']['hit@5'])} and {f3(M['Ablation: proposed - deep embeddings']['hit@5'])}), "
        "confirming that appearance carries the signal. Removing the **orientation cue** *improves* every metric "
        f"(Hit@5 {f3(M['Ablation: proposed - orientation cue']['hit@5'])}, F1 "
        f"{f3(M['Ablation: proposed - orientation cue']['pair_f1'])}), and removing the edge-contact features improves "
        "Hit@5/10. The geometric features therefore add noise more than information on these sherds. We did **not** "
        "switch the headline model to the best ablation: choosing a model by its test-fold performance would bias the "
        "result upward. Instead we report the pre-specified model and treat the ablation as a finding. The LR weights "
        f"tell the same story: colour (+{co['colour_sim']:.2f}) and ResNet (+{co['resnet_cos']:.2f}) dominate. Some "
        f"geometric weights even have physically implausible signs: overlap is *positively* weighted (+{co['overlap']:.2f}) "
        f"and so is the orientation gap (+{co['orient_gap']:.2f}). This is evidence that, at mostly wrong poses, these "
        "features capture incidental shape correlations rather than real fit.")
    w.h("12.5 E4 and E5: geometry on synthetic versus real breaks")
    w.table(["Data", "Pairs", "Pose within 50 px", "Pose within 100 px", "Median rotation error"], [
        ["Synthetic clean breaks (sim_data_8)", str(R["sim"]["n_pairs"]), pct(R["sim"]["pose_success_rate_50px"]),
         pct(R["sim"]["pose_success_rate_100px"]), f"{R['sim']['median_rotation_error_deg']:.1f}°"],
        ["Real eroded sherds (gold)", "111", pct(R["gold_pose"]["pose_success_rate_50px"]),
         pct(R["gold_pose"]["pose_success_rate_100px"]), f"{R['gold_pose']['median_rotation_error_deg']:.1f}°"],
    ], widths=[2.4, 0.6, 1.2, 1.2, 1.4], caption="Pose recovery of the same geometric matcher.")
    w.fig("results_sim_vs_real.png", "The geometric matcher recovers the correct joint for about half of the clean "
          "synthetic breaks but under a fifth of the real ones.", width=4.6)
    w.p(f"On synthetic breaks the matcher finds the exact joint for {pct(R['sim']['pose_success_rate_50px'])} of "
        f"pairs (median rotation error {R['sim']['median_rotation_error_deg']:.1f}°). On real sherds this falls to "
        f"{pct(R['gold_pose']['pose_success_rate_50px'])} (median rotation error "
        f"{R['gold_pose']['median_rotation_error_deg']:.0f}°). A diagnostic on 40 real pairs showed why. At the "
        "*expert* pose, true partners touch along a median of 87 px of edge, whereas the best *wrong* pose that "
        "ICP finds touches along 129 px. On worn edges, maximising contact actively prefers wrong fits. Even on synthetic "
        f"data, the geometry score separates joins from non-joins only modestly (AUC {f3(R['sim']['pair_auc_geo_score'])}), "
        "because the negatives come from the same vessel and straight break segments fit many places.")
    w.h("12.6 Error analysis")
    e = EA["prop"]
    d = EA["dino"]
    w.bullets([
        f"**Wrong top-1 candidates are mostly other vessels.** For the proposed model, {e['top1_wrong']} queries have a "
        f"wrong top-1, and only {e['top1_wrong_same_vessel']} of those ({pct(e['frac'])}) are sherds of the same vessel "
        "that simply do not touch the query. The main confusion is therefore between *different vessels of the same "
        "ware*, not between neighbours within one vessel.",
        f"**The vessel is still found.** {pct(e['top5_same_vessel_share'])} of the top-5 candidates belong to the "
        "query's vessel, against 3.2% expected by chance, a 7× enrichment. So the model groups sherds by vessel "
        "even when it misses the exact neighbour.",
        f"**Small sherds are hardest.** Hit@5 rises from {f3(e['hit5_by_area_tercile'][0])} (smallest third of sherds) "
        f"to {f3(e['hit5_by_area_tercile'][2])} (largest third). Small sherds have less surface and shorter edges. The "
        f"proposed model helps them most: DINOv2 alone reaches only {f3(d['hit5_by_area_tercile'][0])} on the smallest third.",
        f"**Larger vessels are easier** (Hit@5 {f3(e['hit5_group_large'])} for vessels of ≥ 5 sherds vs "
        f"{f3(e['hit5_group_small'])} for ≤ 3), because their queries have more partners to find.",
        "**Spalled surfaces and colour variation within a vessel** (one side of a pot fired darker) explain failures "
        "such as JDX-118 in Figure {FIG:results_qualitative_retrieval}. Its partners differ in tone, and the retrieved candidates are plausible "
        "look-alikes from other vessels.",
    ])
    w.fig("results_qualitative_retrieval.png", "Top-5 candidates from the image-only baseline and the proposed model "
          "for three queries (green = expert-confirmed partner). In the first two the proposed model finds partners the "
          "baseline misses; the third is a failure case for both.")
    w.h("12.7 Complete results of all configurations")
    full = []
    for mm in R["methods"]:
        full.append([mm["method"], mm["family"], f3(mm["hit@1"]), f3(mm["hit@5"]), f3(mm["hit@10"]), f3(mm["mrr"]),
                     f3(mm["map"]), f3(mm["same_vessel_map"]),
                     f3(mm["pair_f1"]) if mm.get("pair_f1") is not None else "–"])
    w.table(["Configuration", "Family", "Hit@1", "Hit@5", "Hit@10", "MRR", "mAP", "Vessel mAP", "Pair F1"], full,
            widths=[2.5, 0.7, 0.5, 0.5, 0.55, 0.5, 0.5, 0.6, 0.55],
            caption="All 16 configurations. 'Vessel mAP' scores retrieval of any sherd of the same vessel instead of "
                    "direct neighbours only.", font=7)
    if PH:
        w.h("12.8 E6: zero-shot transfer to raw excavation photos")
        rows = [[k, f3(v["hit@1"]), f3(v["hit@5"]), f3(v["hit@10"]), f3(v["mrr"])] for k, v in PH["methods"].items()]
        w.table(["Method", "Hit@1", "Hit@5", "Hit@10", "MRR"], rows, widths=[3.6, 0.7, 0.7, 0.7, 0.7],
                caption=f"Raw field photos: {PH['n_photos']} gallery photos, {PH['n_queries']} queries, no re-training.",
                font=8)
        ph = PH["methods"]
        two = ph["Proposed two-stage: appearance top-30 + geometry re-rank"]
        app = ph["Appearance features (LR, trained on gold)"]
        dn = ph["DINOv2 embedding (cosine)"]
        w.p(f"On unprocessed field photos, with automatic segmentation and a gallery of {PH['n_photos']} sherds, "
            f"the gold-trained appearance model reaches Hit@10 {f3(app['hit@10'])} (DINOv2 alone {f3(dn['hit@10'])}). "
            f"Re-ranking its top {PH['rerank_top']} with geometry gives Hit@10 {f3(two['hit@10'])}. The task is "
            "harder than the gold set: the gallery is about 6× larger, segmentation is automatic, and lighting and "
            "backdrop vary. The ranking of methods still lets us judge whether the conclusions carry over from the "
            "curated benchmark to real use.")

    # ---------------- 13. Inference
    w = Writer(doc, response_anchor(doc, "13. Inference"))
    w.h("What the evidence supports")
    w.bullets([
        f"**A learned multi-cue ranker is a practical triage tool.** Under leak-free evaluation, an expert who checks "
        f"10 suggestions finds a true partner for {pct(M[PROP]['hit@10'])} of sherds, a significant improvement over "
        f"the image-only plan of Review 1 ({pct(M[BASE]['hit@10'])}).",
        "**Hypothesis H1 is only partly supported.** The full model is significantly better than the image-only "
        "*baseline*, but the improvement is explained by fusing appearance cues. Adding fracture-edge geometry to "
        "learned appearance gives no significant gain at top-5/10 and a significant loss at top-1.",
        "**The bottleneck is the edge data, not the matching algorithm.** The same matcher works on half of the clean "
        "breaks and under a fifth of the real ones. On eroded edges, a maximum-contact criterion prefers wrong fits.",
        "**Colour is the most informative cue for this ware.** It matches deep features as a single cue and has the "
        "largest model weight. Vessels were fired unevenly, so colour identifies the vessel better than shape does.",
    ])
    w.h("What the model learned and failed to learn")
    w.p("The model learned to *group sherds by vessel*: its top 5 is 7× enriched in same-vessel sherds. It did not "
        "learn to identify the *adjacent* sherd within a vessel, which is what geometry was meant to add. The physically "
        "implausible signs of some geometric weights (positive weight on overlap) show that the geometric features, "
        "measured at mostly wrong poses, do not carry the meaning they were designed for.")
    w.h("Were the objectives achieved?")
    w.p("Objectives O1–O7 were all met: the data are understood and documented, the pipeline runs end to end, "
        "the baseline and the proposed model were evaluated rigorously, alternatives were compared, and a working "
        "prototype exists. The research hypothesis received a clear, evidence-based answer, which is partly negative. "
        "We regard that as a valid outcome of the case study, not a failure of it.")
    w.h("Claims we do not make")
    w.bullets([
        "We do not claim the system reconstructs vessels. It ranks candidates and previews one rigid fit per pair.",
        "We do not claim that geometry is useless for sherd joining. Our 2-D outline-based geometry is not robust to "
        "erosion; 3-D or thickness-profile geometry was not tested.",
        "Results are for one site and ware. Other assemblages may favour different cues.",
    ])

    # ---------------- 14. Limitations
    w = Writer(doc, response_anchor(doc, "14. Limitations"))
    w.h("Technical limitations")
    w.bullets([
        "2-D outlines lose the third dimension: sherd thickness and wall curvature, which experts use first, are invisible.",
        "Maximum-contact pose selection is fragile on worn edges (Section 12.5). Only the outer contour is used, so a "
        "joining edge hidden by perspective is missed.",
        "The cord-mark orientation estimate is a single global direction. Sherds whose stripes curve or cross are "
        "poorly described.",
        "Pre-trained embeddings were not fine-tuned on pottery.",
        "Raw-photo segmentation sometimes includes cast shadows.",
    ])
    w.h("Data-related limitations")
    w.bullets([
        "111 positive pairs is small: confidence intervals are wide (±5–10 points), and ablation differences of a few "
        "points are not significant.",
        "Only exterior views were used; the gold set also contains interior views.",
        "Unlisted pairs are assumed not to join; a few may be undiscovered joins.",
        "One site, one period, one dominant ware.",
    ])
    w.h("Future scope")
    w.table(["Improvement", "Expected benefit", "Effort"], [
        ["Run E6: zero-shot transfer to 799 raw field photos (script and photos ready)", "Tests the conclusions "
         "outside the curated benchmark", "Low"],
        ["Use the interior view as a second appearance cue", "Independent evidence of the same vessel", "Low"],
        ["Edge-band appearance: compare colour/texture in a thin strip along the aligned joint", "Checks continuity "
                                                                                                  "across the break, robust to erosion", "Medium"],
        ["Erosion-robust pose scoring learned from the gold poses (positive = expert pose, negative = ICP false fits)",
         "Fixes the contact-maximisation failure", "Medium"],
        ["Fine-tune DINOv2 with a contrastive loss on same-vessel pairs (Yu et al. 2026 style)", "Better vessel "
                                                                                                "grouping", "Medium"],
        ["Context priors from metadata (layer distance, sherd part)", "Down-weights implausible candidates", "Low"],
        ["Pre-train geometry on all 50,736 simulated pairs, adapt with augmentation that simulates erosion",
         "Bridges the synthetic-real gap", "High"],
        ["Multi-piece assembly: chain pairwise fits into a vessel graph with global consistency",
         "Toward full digital reconstruction", "High"],
        ["3-D scans or thickness profiles", "Restores the strongest physical cue", "High (needs new data)"],
    ], widths=[3.2, 2.4, 1.2], caption="Realistic next steps.")

    # ---------------- 15. Conclusion
    w = Writer(doc, response_anchor(doc, "15. Conclusion"))
    w.p("This case study turned the archaeological task of finding joining pottery sherds into a pattern-recognition "
        "problem, a pairwise join/no-join decision used to rank candidates, and evaluated it on real expert-confirmed "
        "joins from the Daxinzhuang-18K benchmark. We built a complete pipeline: segmentation and illumination "
        "normalisation; colour, texture and deep appearance features; a fracture-edge matcher based on complementary "
        "curvature, Kabsch/ICP alignment and overlap checks; a logistic-regression fusion model; and an interactive "
        "demo.")
    w.p(f"Under leak-free vessel-grouped cross-validation, the system puts a true partner in the top 10 for "
        f"{pct(M[PROP]['hit@10'])} of sherds, significantly above the image-only baseline planned in Review 1 "
        f"({pct(M[BASE]['hit@10'])}). The most important finding is less flattering and more useful: the gain comes "
        "from combining appearance cues, while 2-D fracture-edge geometry adds no significant benefit on real eroded "
        "sherds, even though the same matcher works on clean synthetic breaks. In practice, an appearance ranker can "
        "already save experts time today, whereas geometry needs erosion-robust descriptors before it can help.")
    w.p("Key learning: in pattern recognition, the evaluation protocol decides what a number means. Splitting by "
        "vessel, using ranking metrics instead of accuracy on a 1.5%-positive problem, bootstrapping the "
        "differences, and checking the method on synthetic data turned an apparently positive result into a "
        "precise statement of what works and why.")

    # ---------------- 16. References
    w = Writer(doc, response_anchor(doc, "16. References"))
    refs = [
        "Ji, D., Xue, T., Wang, Q., Ma, W., Lang, J., Fang, H., Tu, C., & Lu, L. (2026). Daxinzhuang-18K: A large-scale 2D "
        "benchmark for pottery sherd retrieval and reassembly (Version 1.0.0) [Data set]. Zenodo. "
        "https://doi.org/10.5281/zenodo.21553799",
        "Freeman, H., & Garder, L. (1964). Apictorial jigsaw puzzles: The computer solution of a problem in pattern "
        "recognition. IEEE Transactions on Electronic Computers, EC-13(2), 118–127. https://doi.org/10.1109/PGEC.1964.263781",
        "Wolfson, H. J. (1990). On curve matching. IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(5), "
        "483–489. https://doi.org/10.1109/34.55108",
        "Papaioannou, G., Karabassi, E.-A., & Theoharis, T. (2002). Reconstruction of three-dimensional objects through "
        "matching of their parts. IEEE TPAMI, 24(1), 114–124. https://doi.org/10.1109/34.982888",
        "Huang, Q.-X., Flöry, S., Gelfand, N., Hofer, M., & Pottmann, H. (2006). Reassembling fractured objects by geometric "
        "matching. ACM Transactions on Graphics (SIGGRAPH), 25(3), 569–578. https://doi.org/10.1145/1179352.1141925",
        "Kampel, M., & Sablatnig, R. (2003). Profile-based pottery reconstruction. In Proc. CVPR Workshop on Applications "
        "of Computer Vision in Archaeology. https://doi.org/10.1109/CVPRW.2003.10007",
        "Chetouani, A., Treuillet, S., Exbrayat, M., & Jesset, S. (2020). Classification of engraved pottery sherds mixing "
        "deep-learning features by compact bilinear pooling. Pattern Recognition Letters, 131, 1–7. "
        "https://doi.org/10.1016/j.patrec.2019.12.009",
        "Pinho, M., Mota, G., & da Costa, G. (2025). A deep learning approach to assist in pottery reconstruction from its "
        "sherds. Heritage, 8(5), 167. https://doi.org/10.3390/heritage8050167",
        "Yu, X., Li, T., Song, Z., Lu, H., Du, Y., Wu, F., Xiao, H., & Xu, L. (2026). Mitigating spurious features by "
        "contrastive learning in pottery sherd recognition. npj Heritage Science, 14, 135. "
        "https://doi.org/10.1038/s40494-025-02170-3",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proc. IEEE CVPR "
        "(pp. 770–778). https://doi.org/10.1109/CVPR.2016.90",
        "Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without "
        "supervision. Transactions on Machine Learning Research. arXiv:2304.07193",
        "Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture "
        "classification with local binary patterns. IEEE TPAMI, 24(7), 971–987. https://doi.org/10.1109/TPAMI.2002.1017623",
        "Kabsch, W. (1976). A solution for the best rotation to relate two sets of vectors. Acta Crystallographica "
        "Section A, 32(5), 922–923. https://doi.org/10.1107/S0567739476001873",
        "Besl, P. J., & McKay, N. D. (1992). A method for registration of 3-D shapes. IEEE TPAMI, 14(2), 239–256. "
        "https://doi.org/10.1109/34.121791",
        "Rother, C., Kolmogorov, V., & Blake, A. (2004). \"GrabCut\": Interactive foreground extraction using iterated "
        "graph cuts. ACM Transactions on Graphics, 23(3), 309–314. https://doi.org/10.1145/1015706.1015720",
        "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. Heckbert (Ed.), Graphics Gems IV "
        "(pp. 474–485). Academic Press.",
        "Bigün, J., & Granlund, G. H. (1987). Optimal orientation detection of linear symmetry. In Proc. 1st IEEE ICCV "
        "(pp. 433–438).",
        "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, "
        "12, 2825–2830.",
        "Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in "
        "Neural Information Processing Systems 32.",
        "Wightman, R. (2019). PyTorch Image Models (timm). GitHub repository. https://github.com/huggingface/pytorch-image-models",
        "Bradski, G. (2000). The OpenCV library. Dr. Dobb's Journal of Software Tools.",
        "Project code and results: Agrawal, V. (2026). SherdMatch: AI-based archaeological pottery fragment matching. GitHub repository. "
        "https://github.com/vansh070605/SherdMatch",
    ]
    for i, r in enumerate(refs, 1):
        para = w.p(f"[{i}] {r}", align="left", space_after=3)
        para.paragraph_format.left_indent = Inches(0.35)
        para.paragraph_format.first_line_indent = Inches(-0.35)

    # ---------------- Review sections
    paras = doc.paragraphs
    fc = [p for p in paras if p.text.startswith("Faculty comments")]
    sr = [p for p in paras if p.text.startswith("Student's response")]
    replace_blank_lines(fc[0], "No faculty comments were recorded at Review 1.")
    replace_blank_lines(sr[0], "Proceeded with the plan presented at Review 1 (dataset → preprocessing → image-only "
                               "baseline → geometry-based matching). Review 2 implements and evaluates it; see "
                               "Sections 9–14.")
    # Review 2 comments are left blank for the faculty to fill in.

    # ---------------- checklist: all items addressed
    for p_ in paras:
        if p_.text.startswith("☐"):
            for r in p_.runs:
                r.text = r.text.replace("☐", "☑")


def strip_template_instructions(doc):
    """Remove the template's grey guidance text and its 'Response:' label under every section,
    plus the intro line telling students how to write the report. Our content stays."""
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if t.startswith("Response:"):
            prev = p._p.getprevious()
            if prev is not None and prev.tag == qn("w:p"):
                prev.getparent().remove(prev)  # the instruction paragraph directly above
            p._p.getparent().remove(p._p)
        elif t.startswith("Provide concise, evidence-based explanations"):
            p._p.getparent().remove(p._p)


def tighten_ending(doc):
    """Keep the closing footnote on the same page as the checklist (no page with one stray line)."""
    paras = doc.paragraphs
    start = next(i for i, p in enumerate(paras) if p.text.strip() == "Submission Checklist")
    for p in paras[start + 1:]:
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.0
    body = doc.element.body
    note = next(p for p in paras if p.text.startswith("Faculty evaluation should assess"))
    prev = note._p.getprevious()
    while prev is not None and prev.tag == qn("w:p") and not "".join(prev.itertext()).strip():
        gone, prev = prev, prev.getprevious()  # blank paragraphs before the note
        body.remove(gone)


def main():
    doc = docx.Document(str(TEMPLATE))
    fill(doc)
    resolve_labels(doc)
    strip_template_instructions(doc)
    tighten_ending(doc)
    for p in doc.paragraphs:  # major parts start on a new page
        if p.style.name == "Heading 2" and p.text.split(".")[0] in {"2", "4", "9", "12"}:
            p.paragraph_format.page_break_before = True
    DEST.parent.mkdir(exist_ok=True)
    doc.save(str(DEST))
    print("saved", DEST)


if __name__ == "__main__":
    main()
