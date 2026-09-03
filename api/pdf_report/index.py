"""
CareerCast — /api/pdf_report
Generates a PDF analysis report for a single prediction result and
streams it back as an attachment.  Requires reportlab.
"""

import io
import os
import sys

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/api/pdf_report", methods=["POST"])
def pdf_report():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    required = ("top_match", "top_score", "ranked_jobs", "skills", "preview")
    for key in required:
        if key not in data:
            return jsonify({"error": f"Missing field: {key}"}), 400

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER

        buf   = io.BytesIO()
        doc   = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm,  bottomMargin=2*cm,
        )
        styles = getSampleStyleSheet()
        brand  = colors.HexColor("#1f3a2e")

        T  = ParagraphStyle("T", parent=styles["Title"],
                            fontSize=20, textColor=brand,
                            alignment=TA_CENTER, spaceAfter=4)
        H  = ParagraphStyle("H", parent=styles["Heading2"],
                            fontSize=12, textColor=brand, spaceAfter=3)
        B  = ParagraphStyle("B", parent=styles["Normal"],
                            fontSize=10, leading=14)
        S  = ParagraphStyle("S", parent=styles["Normal"],
                            fontSize=8, textColor=colors.HexColor("#778870"))
        FC = ParagraphStyle("FC", parent=S, alignment=TA_CENTER)

        story = []
        story.append(Paragraph("CareerCast Analysis Report", T))
        story.append(Paragraph(f"Resume: {data.get('resume_name', 'Resume')}", S))
        story.append(HRFlowable(width="100%", thickness=1, color=brand, spaceAfter=10))

        # Top match
        story.append(Paragraph("Top Match", H))
        mt = Table(
            [["Category", "Confidence"],
             [data["top_match"], f"{data['top_score']:.1f}%"]],
            colWidths=[11*cm, 5*cm],
        )
        mt.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0), brand),
            ("TEXTCOLOR",    (0,0),(-1,0), colors.white),
            ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
            ("BACKGROUND",   (0,1),(-1,1), colors.HexColor("#e8f0e9")),
            ("FONTNAME",     (0,1),(-1,1), "Helvetica-Bold"),
            ("FONTSIZE",     (0,1),(-1,1), 11),
            ("ALIGN",        (1,0),(1,-1), "CENTER"),
            ("GRID",         (0,0),(-1,-1), 0.3, colors.HexColor("#dde0d8")),
            ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#c8dbca")),
            ("TOPPADDING",   (0,0),(-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ]))
        story.append(mt)
        story.append(Spacer(1, 12))

        # Ranked categories
        story.append(Paragraph("Ranked Categories", H))
        rd = [["#", "Category", "Score"]]
        for i, r in enumerate(data["ranked_jobs"], 1):
            rd.append([str(i), r["job"], f"{r['score']:.1f}%"])
        rt = Table(rd, colWidths=[1.5*cm, 10*cm, 4.5*cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0), brand),
            ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
            ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 9),
            ("ALIGN",          (0,0),(0,-1), "CENTER"),
            ("ALIGN",          (2,0),(2,-1), "CENTER"),
            ("ROWBACKGROUNDS", (0,1),(-1,-1),
             [colors.white, colors.HexColor("#f4f7f4")]),
            ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor("#dde0d8")),
            ("BOX",            (0,0),(-1,-1), 0.5, colors.HexColor("#c8dbca")),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
        ]))
        story.append(rt)
        story.append(Spacer(1, 12))

        # Skills
        story.append(Paragraph("Skills Found", H))
        skills_text = ", ".join(data["skills"]) if data["skills"] else "None detected."
        story.append(Paragraph(skills_text, B))
        story.append(Spacer(1, 8))

        if data.get("skills_suggested"):
            story.append(Paragraph(f"Skill Gaps for '{data['top_match']}'", H))
            story.append(Paragraph(", ".join(data["skills_suggested"]), B))
            story.append(Spacer(1, 12))

        # Preview
        if data.get("preview"):
            story.append(Paragraph("Resume Text Preview", H))
            preview = (data["preview"][:500] + "…") if len(data["preview"]) > 500 \
                      else data["preview"]
            prev_s = ParagraphStyle(
                "P", parent=B,
                backColor=colors.HexColor("#f9faf7"),
                borderPadding=6, fontSize=8.5, leading=13,
            )
            story.append(Paragraph(preview.replace("\n", "<br/>"), prev_s))

        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#aab4a2"), spaceAfter=5))
        story.append(Paragraph(
            "Generated by CareerCast — AI-Powered Resume Intelligence Platform",
            FC,
        ))

        doc.build(story)
        buf.seek(0)

        safe_name = (data.get("resume_name") or "report").replace(" ", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"careercast_{safe_name}.pdf",
        )

    except ImportError:
        return jsonify({
            "error": "reportlab is not installed. Run: pip install reportlab"
        }), 500
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"PDF generation failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)
