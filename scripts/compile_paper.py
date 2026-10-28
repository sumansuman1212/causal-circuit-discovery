import asyncio
import html
import os
import re
import sys
from pathlib import Path
 
import markdown
 
ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
MD_PATH = PAPER_DIR / "paper.md"
HTML_PATH = PAPER_DIR / "paper.html"
PDF_PATH = PAPER_DIR / "paper.pdf"
 
MATHJAX_SRC = os.environ.get(
    "MATHJAX_SRC", "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
)
 
# --------------------------------------------------------------------------
# 1. Math protection
# --------------------------------------------------------------------------
DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)")
 
 
def protect_math(text):
    """Swap every math span for an inert token; return (text, store)."""
    store = []
 
    def stash(kind):
        def _sub(m):
            store.append((kind, m.group(1)))
            return f"MATHTOKEN{len(store) - 1}END"
        return _sub
 
    text = DISPLAY_RE.sub(stash("display"), text)
    text = INLINE_RE.sub(stash("inline"), text)
    return text, store
 
 
def restore_math(page, store):
    for i, (kind, body) in enumerate(store):
        safe = html.escape(body, quote=False)  # keeps < > & valid in HTML
        if kind == "display":
            repl = f'<div class="math-display">$${safe}$$</div>'
            # a display token alone in a <p> must not stay inside the <p>
            page = page.replace(f"<p>MATHTOKEN{i}END</p>", repl)
        else:
            repl = f"${safe}$"
        page = page.replace(f"MATHTOKEN{i}END", repl)
    return page
 
 
# --------------------------------------------------------------------------
# 2. Blank line before lists (python-markdown requirement)
# --------------------------------------------------------------------------
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
 
 
def ensure_blank_before_lists(text):
    out, prev = [], ""
    for line in text.split("\n"):
        if (
            LIST_ITEM.match(line)
            and prev.strip()
            and not LIST_ITEM.match(prev)
            and not prev.startswith((" ", "\t"))      # nested / continuation
            and not prev.lstrip().startswith(("|", "#", ">"))
        ):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)
 
 
# --------------------------------------------------------------------------
# 3. Figures
# --------------------------------------------------------------------------
FIGURE_RE = re.compile(
    r"<p>\s*(<img[^>]*/?>)\s*(?:<br\s*/?>)?\s*<em>(.*?)</em>\s*</p>", re.DOTALL
)
 
 
def wrap_figures(page):
    return FIGURE_RE.sub(
        r"<figure>\1<figcaption>\2</figcaption></figure>", page
    )
 
 
# --------------------------------------------------------------------------
# CSS / HTML shell
# --------------------------------------------------------------------------
CSS = """
@page { size: A4; margin: 20mm 18mm 22mm 18mm; }
 
body {
  font-family: 'Times New Roman', Times, Georgia, serif;
  font-size: 10.5pt; line-height: 1.55; color: #111;
  max-width: 820px; margin: 0 auto; padding: 10px; background: #fff;
}
@media print { body { max-width: none; padding: 0; } }
 
h1 {
  font-size: 19pt; font-weight: 700; text-align: center;
  margin: 10px 0 12px; line-height: 1.3; color: #0b1a30;
}
/* centre ONLY the title-block paragraph, not every paragraph with bold text */
h1 + p { text-align: center; }
h1 + p strong:first-child { font-size: 10.5pt; }
 
hr { border: 0; border-top: 1px solid #ccc; margin: 18px 0; }
 
h2 {
  font-size: 13.5pt; font-weight: 700; color: #0b1a30;
  margin: 22px 0 8px; border-bottom: 1px solid #e0e0e0; padding-bottom: 3px;
}
h3 { font-size: 11.5pt; font-weight: 700; color: #1a2a40; margin: 16px 0 6px; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
 
/* Abstract box: first paragraph after the Abstract heading */
h2#abstract + p {
  background: #f8f9fa; border-left: 3px solid #2b5c8f;
  padding: 12px 16px; margin: 12px 0 10px; font-size: 10pt;
  line-height: 1.5; text-align: justify;
}
h2#abstract ~ p:not(:has(+ h2)):not(:has(+ hr)) { text-align: justify; }
 
p { text-align: justify; margin: 8px 0; }
ul, ol { margin: 8px 0 12px 24px; padding: 0; }
li { margin-bottom: 5px; text-align: justify; }
 
/* Tables */
table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 9.5pt;
        break-inside: avoid; page-break-inside: avoid; }
th, td { padding: 7px 10px; text-align: center; }
th { border-top: 1.5px solid #111; border-bottom: 1.5px solid #111;
     font-weight: 700; background: #f5f6f8; }
td { border-bottom: 1px solid #e2e4e8; }
tbody tr:last-child td { border-bottom: 1.5px solid #111; }
 
/* Figures */
figure { margin: 14px 0; text-align: center;
         break-inside: avoid; page-break-inside: avoid; }
figure img { display: block; max-width: 92%; height: auto; margin: 0 auto 6px;
             border: 1px solid #e5e5e5; border-radius: 4px; }
figcaption { font-style: italic; font-size: 9pt; color: #444;
             text-align: center; margin: 4px 8px 14px; }
 
/* em stays INLINE everywhere else */
em { font-style: italic; }
 
code { font-family: Consolas, 'Courier New', monospace; font-size: 9.5pt;
       background: #f4f5f7; padding: 2px 4px; border-radius: 3px; }
td code { white-space: nowrap; }
 
/* Math */
.math-display { margin: 10px 0; overflow-x: auto; overflow-y: hidden;
                break-inside: avoid; page-break-inside: avoid; }
mjx-container[jax="SVG"] { max-width: 100%; }
mjx-container[jax="SVG"][display="true"] { margin: 0.6em 0 !important; }
"""
 
SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script id="MathJax-script" async src="{mathjax}"></script>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""
 
 
def build_html():
    if not MD_PATH.exists():
        sys.exit(f"Cannot find {MD_PATH}")
 
    text = MD_PATH.read_text(encoding="utf-8")
 
    text, store = protect_math(text)
    text = ensure_blank_before_lists(text)
 
    body = markdown.markdown(
        text,
        extensions=["tables", "sane_lists", "toc"],
        output_format="html5",
    )
    body = wrap_figures(body)
    body = restore_math(body, store)
 
    leftovers = re.findall(r"MATHTOKEN\d+END", body)
    if leftovers:
        sys.exit(f"Unrestored math tokens: {leftovers[:5]}")
 
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.DOTALL)
    title = html.escape(re.sub(r"<[^>]+>", "", m.group(1))) if m else "Paper"
 
    HTML_PATH.write_text(
        SHELL.format(title=title, mathjax=MATHJAX_SRC, css=CSS, body=body),
        encoding="utf-8",
    )
 
    # warn about missing figures (paths are relative to paper/)
    for src in re.findall(r'<img[^>]+src="([^"]+)"', body):
        if not (PAPER_DIR / src).resolve().exists():
            print(f"WARNING: figure not found: {src}")
    print(f"Wrote {HTML_PATH}")
 
 
# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
FOOTER = (
    '<div style="width:100%;font-size:9px;font-family:Times New Roman,serif;'
    'color:#555;text-align:center;"><span class="pageNumber"></span></div>'
)
 
 
async def print_pdf():
    from playwright.async_api import async_playwright
 
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(HTML_PATH.resolve().as_uri(), wait_until="networkidle")
 
        # wait for MathJax to finish, then for every image to decode
        await page.wait_for_function(
            "window.MathJax && MathJax.startup && MathJax.startup.promise"
        )
        await page.evaluate("MathJax.startup.promise.then(() => MathJax.typesetPromise())")
        await page.evaluate(
            "Promise.all([...document.images].map(i => i.complete ? 0 : "
            "new Promise(r => { i.onload = i.onerror = r; })))"
        )
        await page.evaluate("document.fonts.ready")
 
        bad = await page.evaluate(
            "document.querySelectorAll('mjx-merror').length"
        )
        if bad:
            print(f"WARNING: {bad} MathJax error(s) in document")
 
        await page.pdf(
            path=str(PDF_PATH),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=FOOTER,
            margin={"top": "20mm", "bottom": "22mm", "left": "18mm", "right": "18mm"},
        )
        await browser.close()
    print(f"Wrote {PDF_PATH}")
 
 
if __name__ == "__main__":
    build_html()
    asyncio.run(print_pdf())