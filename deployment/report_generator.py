from datetime import date
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#173F63")
TEAL = colors.HexColor("#1F7F95")
PALE_BLUE = colors.HexColor("#EAF2F7")
PALE_TEAL = colors.HexColor("#E9F6F7")
LIGHT_BORDER = colors.HexColor("#C7D5DF")
DARK_TEXT = colors.HexColor("#1C2630")
MUTED_TEXT = colors.HexColor("#596773")
WARNING_BG = colors.HexColor("#FFF4E5")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12,
            textColor=MUTED_TEXT,
            alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            textColor=DARK_TEXT,
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=MUTED_TEXT,
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=DARK_TEXT,
        ),
        "table_bold": ParagraphStyle(
            "TableBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=DARK_TEXT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=NAVY,
            spaceBefore=3,
        ),
    }


def _p(text, style):
    return Paragraph(str(text), style)


def _section(number, title, styles):
    return _p(f"{number}. {title}", styles["section"])


def _field_table(rows, styles):
    data = []
    for row in rows:
        data.append([
            _p(row[0], styles["table_bold"]),
            _p(row[1], styles["table"]),
            _p(row[2], styles["table_bold"]),
            _p(row[3], styles["table"]),
        ])
    table = Table(data, colWidths=[28 * mm, 56 * mm, 30 * mm, 58 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
        ("BACKGROUND", (2, 0), (2, -1), PALE_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _attention_png(image, cam):
    image = np.asarray(image, dtype=np.float32)
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    support = image > 1e-5

    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=220)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    masked_cam = np.ma.masked_where(~support, cam)
    ax.imshow(
        masked_cam,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=np.where(support, 0.55, 0.0),
        interpolation="lanczos",
    )
    if np.any((cam >= 0.60) & support):
        try:
            ax.contour(np.ma.masked_where(~support, cam), levels=[0.60], colors="black", linewidths=0.8)
        except Exception:
            pass
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output = BytesIO()
    fig.savefig(output, format="png", dpi=220, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    output.seek(0)
    return output


def _location_phrase(location):
    horizontal, vertical = location.split("-", 1)
    return f"the {vertical} {horizontal} part of the displayed image"


def _clinical_phrase(feature, value, predicted_name):
    value = float(value)
    if value > 0.01:
        return f"{feature} gave a small amount of additional support to the {predicted_name} result."
    if value < -0.01:
        return f"{feature} slightly reduced the model's support for the {predicted_name} result."
    return f"{feature} had very little effect on the {predicted_name} result."


def _footer(canvas, document):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LIGHT_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 14 * mm, width - 20 * mm, 14 * mm)
    canvas.setFillColor(MUTED_TEXT)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(20 * mm, 9 * mm, "AI system: Multimodal MRI + Clinical Data | Grad-CAM + Shapley explanations")
    canvas.drawRightString(width - 20 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_patient_report(
    age,
    sex,
    predicted_name,
    confidence,
    probabilities,
    processed_views,
    gradcam,
    cam_summaries,
    age_shapley,
    sex_shapley,
):
    """Return a polished, print-ready AI assessment report as PDF bytes."""
    styles = _styles()
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="AI-Assisted Brain Tumor Assessment Report",
        author="Brain Tumor XAI",
    )

    confidence_percent = float(confidence) * 100.0
    predicted_title = str(predicted_name).title()
    report_date = date.today().strftime("%d/%m/%Y")
    story = [
        _p("AI-ASSISTED BRAIN TUMOR ASSESSMENT REPORT", styles["title"]),
        _p("Clinician-Facing and Patient-Friendly Explainable AI Report", styles["subtitle"]),
        Spacer(1, 5 * mm),
        _section(1, "Patient Information", styles),
        _field_table([
            ("Report Date", report_date, "Study Type", "Brain MRI"),
            ("Age", f"{float(age):.0f} years", "Sex/Gender", str(sex).title()),
            ("MRI Views", "Axial + Coronal + Sagittal", "Assessment", "AI-assisted"),
        ], styles),
        _section(2, "AI Prediction", styles),
        _field_table([
            ("Predicted Type", predicted_title, "Confidence", f"{confidence_percent:.2f}%"),
        ], styles),
        Spacer(1, 2 * mm),
    ]

    probability_rows = [[
        _p("Tumor category", styles["table_header"]),
        _p("Model probability", styles["table_header"]),
    ]]
    for class_name, probability in probabilities.items():
        probability_rows.append([
            _p(str(class_name).title(), styles["table"]),
            _p(f"{float(probability) * 100.0:.2f}%", styles["table"]),
        ])
    probability_table = Table(probability_rows, colWidths=[115 * mm, 57 * mm])
    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]))
    story.extend([probability_table, _section(3, "MRI Analysis", styles)])

    analysis_rows = [[
        _p("MRI View", styles["table_header"]),
        _p("AI-Assisted Analysis", styles["table_header"]),
    ]]
    for view_name in ("axial", "coronal", "sagittal"):
        location = _location_phrase(cam_summaries[view_name]["location"])
        analysis_rows.append([
            _p(view_name.title(), styles["table_bold"]),
            _p(
                f"The model's strongest attention was in {location}. This describes model attention "
                "in image coordinates and is not a confirmed anatomical finding or tumor boundary.",
                styles["table"],
            ),
        ])
    analysis_table = Table(analysis_rows, colWidths=[35 * mm, 137 * mm], repeatRows=1)
    analysis_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([analysis_table, PageBreak(), _section(4, "Explainable AI (XAI) Findings", styles)])

    image_cells = []
    caption_cells = []
    for view_name in ("axial", "coronal", "sagittal"):
        image_data = _attention_png(processed_views[view_name], gradcam[view_name])
        image_cells.append(Image(image_data, width=52 * mm, height=52 * mm))
        caption_cells.append(_p(f"{view_name.title()} attention", styles["caption"]))
    image_table = Table([image_cells, caption_cells], colWidths=[57.3 * mm] * 3)
    image_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.extend([
        image_table,
        Spacer(1, 2 * mm),
        _p(
            "Grad-CAM highlights the image regions that influenced the classification. Red and yellow indicate "
            "stronger model attention; blue indicates lower attention. These overlays are explanations of AI "
            "behavior, not lesion segmentations or confirmed tumor outlines.",
            styles["small"],
        ),
        _section(5, "Natural-Language Clinical Explanation", styles),
        _p(
            f"The AI-assisted review found image patterns most consistent with <b>{predicted_title}</b>, with "
            f"a confidence score of <b>{confidence_percent:.2f}%</b>. The system considered the axial, coronal, "
            "and sagittal MRI images together. The colored overlays show the areas that contributed most to the "
            "model's choice; they do not by themselves prove where a tumor is located.",
            styles["body"],
        ),
        Spacer(1, 2 * mm),
        _p(
            _clinical_phrase("The patient's age", age_shapley, predicted_title) + " " +
            _clinical_phrase("The recorded sex", sex_shapley, predicted_title) +
            " The MRI images remained the main source of information for the prediction.",
            styles["body"],
        ),
        _section(6, "Patient-Friendly Summary", styles),
        _p(
            f"In simple terms, the AI found that the scan looks most like the <b>{predicted_title}</b> category. "
            "This is a computer-generated assessment, not a final diagnosis. A radiologist or treating specialist "
            "should review the complete original MRI study, the patient's symptoms and medical history, and any "
            "other required tests before deciding what the finding means or what care is needed.",
            styles["body"],
        ),
        _section(7, "Suggested Next Steps", styles),
        _p(
            "1. Share the complete original MRI study with a qualified radiologist.<br/>"
            "2. Compare with earlier scans when available.<br/>"
            "3. Discuss whether contrast imaging, follow-up imaging, or other tests are appropriate.<br/>"
            "4. Do not start, stop, or change treatment based only on this report.",
            styles["body"],
        ),
        _section(8, "Clinical Disclaimer", styles),
    ])

    disclaimer = Table([[
        _p(
            "<b>Important:</b> This report represents an AI-assisted research and decision-support assessment. "
            "It is not a definitive medical diagnosis and does not replace evaluation by a qualified radiologist, "
            "neurologist, neurosurgeon, oncologist, or other medical professional. Final diagnosis and treatment "
            "decisions must be based on the complete clinical evaluation, radiological findings, and appropriate tests.",
            styles["body"],
        )
    ]], colWidths=[172 * mm])
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARNING_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D79B3B")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(KeepTogether(disclaimer))

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    output.seek(0)
    return output.getvalue()
