#!/usr/bin/env python3
"""
Compiles paper/paper.md into a publication-grade PDF using headless browser rendering.
Generates paper/paper.html with MathJax and publication CSS, then prints to paper/paper.pdf.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

import markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_MD = REPO_ROOT / "paper" / "paper.md"
PAPER_HTML = REPO_ROOT / "paper" / "paper.html"
PAPER_PDF = REPO_ROOT / "paper" / "paper.pdf"

EDGE_PATH = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
CHROME_PATH = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")


def get_browser_executable() -> Path:
    if EDGE_PATH.exists():
        return EDGE_PATH
    if CHROME_PATH.exists():
        return CHROME_PATH
    raise FileNotFoundError("Neither Microsoft Edge nor Google Chrome executable found for PDF printing.")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Automated Discovery and Causal Validation of Minimal Transformer Circuits</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  svg: {{
    fontCache: 'global'
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
@page {{
  size: A4;
  margin: 20mm 18mm 20mm 18mm;
  @bottom-center {{
    content: counter(page);
    font-size: 9pt;
    font-family: 'Times New Roman', Times, serif;
    color: #555;
  }}
}}

body {{
  font-family: 'Times New Roman', Times, Georgia, serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #111;
  max-width: 820px;
  margin: 0 auto;
  padding: 10px;
  background-color: #fff;
}}

h1:first-of-type {{
  font-size: 19pt;
  font-weight: 700;
  text-align: center;
  margin-top: 10px;
  margin-bottom: 12px;
  line-height: 1.3;
  color: #0b1a30;
}}

/* Authors & Metadata */
p:has(strong:contains("Author")), p:has(strong) {{
  text-align: center;
}}

p > strong:first-child {{
  font-size: 10.5pt;
}}

hr {{
  border: 0;
  border-top: 1px solid #ccc;
  margin: 18px 0;
}}

h2 {{
  font-size: 13.5pt;
  font-weight: 700;
  color: #0b1a30;
  margin-top: 22px;
  margin-bottom: 8px;
  border-bottom: 1px solid #e0e0e0;
  padding-bottom: 3px;
}}

h3 {{
  font-size: 11.5pt;
  font-weight: 700;
  color: #1a2a40;
  margin-top: 16px;
  margin-bottom: 6px;
}}

/* Abstract Styling */
h2#abstract + p,
h2:first-of-type + p {{
  background: #f8f9fa;
  border-left: 3px solid #2b5c8f;
  padding: 12px 16px;
  margin: 12px 0 20px 0;
  font-size: 10pt;
  line-height: 1.5;
  text-align: justify;
}}

p {{
  text-align: justify;
  margin: 8px 0;
}}

ul, ol {{
  margin: 8px 0 12px 24px;
  padding: 0;
}}

li {{
  margin-bottom: 5px;
}}

/* Tables: Booktabs Academic Style */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
  font-size: 9.5pt;
}}

th, td {{
  padding: 7px 10px;
  text-align: center;
}}

th {{
  border-top: 1.5px solid #111;
  border-bottom: 1.5px solid #111;
  font-weight: 700;
  background-color: #f5f6f8;
}}

td {{
  border-bottom: 1px solid #e2e4e8;
}}

tbody tr:last-child td {{
  border-bottom: 1.5px solid #111;
}}

/* Figures & Images */
img {{
  display: block;
  max-width: 90%;
  height: auto;
  margin: 14px auto 6px auto;
  border: 1px solid #e5e5e5;
  border-radius: 4px;
}}

em {{
  display: block;
  text-align: center;
  font-size: 9pt;
  color: #444;
  margin-bottom: 14px;
}}

/* Code & Monospace */
code {{
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 9.5pt;
  background-color: #f4f5f7;
  padding: 2px 4px;
  border-radius: 3px;
}}

blockquote {{
  margin: 14px 0;
  padding: 8px 14px;
  background-color: #f0f4f8;
  border-left: 3px solid #2b5c8f;
  font-size: 9.5pt;
}}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def compile_markdown_to_pdf():
    if not PAPER_MD.exists():
        raise FileNotFoundError(f"Markdown paper not found at {PAPER_MD}")

    print(f"Reading {PAPER_MD}...")
    with open(PAPER_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert relative image paths to file:// URIs so Edge can resolve them
    results_fig_dir = (REPO_ROOT / "results" / "figures").resolve()
    md_text = md_text.replace("../results/figures/", f"{results_fig_dir.as_uri()}/")

    print("Converting Markdown to HTML...")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )

    full_html = HTML_TEMPLATE.format(content=html_body)

    with open(PAPER_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved intermediate HTML to {PAPER_HTML}")

    # Launch headless Edge to print to PDF
    browser_exe = get_browser_executable()
    html_uri = PAPER_HTML.resolve().as_uri()
    print(f"Printing to PDF using {browser_exe.name}...")

    cmd = [
        str(browser_exe),
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={str(PAPER_PDF.resolve())}",
        html_uri,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"Browser stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Browser printing failed with code {result.returncode}")

    # Give OS a moment to flush file buffer
    time.sleep(1)

    if PAPER_PDF.exists() and PAPER_PDF.stat().st_size > 0:
        size_kb = PAPER_PDF.stat().st_size / 1024
        print(f"\n[SUCCESS] Successfully generated publication PDF: {PAPER_PDF} ({size_kb:.1f} KB)")
    else:
        raise RuntimeError(f"Output PDF {PAPER_PDF} was not created or is empty.")


if __name__ == "__main__":
    compile_markdown_to_pdf()
