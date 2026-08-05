from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape

INPUT = "swahili-sentiment-report.md"
OUTPUT = "swahili-sentiment-report.pdf"

style_sheet = getSampleStyleSheet()
style_sheet["Heading1"].spaceAfter = 12
style_sheet["Heading1"].spaceBefore = 12
style_sheet["Heading2"].spaceAfter = 10
style_sheet["Heading2"].spaceBefore = 8
style_sheet["BodyText"].spaceAfter = 6
style_sheet["BodyText"].leading = 14
style_sheet.add(ParagraphStyle(name="CustomBullet", parent=style_sheet["BodyText"], leftIndent=12, bulletIndent=0, spaceAfter=2, leading=14))


def parse_markdown(lines):
    flowables = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            flowables.append(Spacer(1, 0.12 * inch))
            continue
        if line.startswith("# "):
            text = escape(line[2:].strip())
            flowables.append(Paragraph(f"<b>{text}</b>", style_sheet["Heading1"]))
        elif line.startswith("## "):
            text = escape(line[3:].strip())
            flowables.append(Paragraph(f"<b>{text}</b>", style_sheet["Heading2"]))
        elif line.startswith("### "):
            text = escape(line[4:].strip())
            flowables.append(Paragraph(f"<b>{text}</b>", style_sheet["Heading3"]))
        elif line.startswith("- "):
            text = escape(line[2:].strip())
            flowables.append(Paragraph(f"• {text}", style_sheet["CustomBullet"]))
        else:
            flowables.append(Paragraph(escape(line), style_sheet["BodyText"]))
    return flowables


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = SimpleDocTemplate(OUTPUT, pagesize=letter, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, topMargin=0.75 * inch,
                            bottomMargin=0.75 * inch)
    story = parse_markdown(lines)
    doc.build(story)
    print(f"Generated PDF: {OUTPUT}")


if __name__ == "__main__":
    main()
