"""Build the paper flat and report content pages against the workshop limit.

    python scripts/check_pages.py

The workshop allows 8 pages excluding references AND appendices, so "content pages" means
everything before the References heading. Exits non-zero if that count goes over.
Run with the system python: PyMuPDF lives in ~/.local and PYTHONNOUSERSITE=1 hides it.
"""
from __future__ import annotations
import os, re, shutil, subprocess, sys, tempfile

LIMIT = 8
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(tmp):
    for d in ("figures", "tables"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
        for f in os.listdir(os.path.join(HERE, d)):
            shutil.copy(os.path.join(HERE, d, f), os.path.join(tmp, d, f))
    src = open(os.path.join(HERE, "paper", "main.tex")).read()
    # the paper keeps figures and tables a level up; flatten those paths for the build
    src = src.replace("{{../figures/}}", "{{figures/}}").replace("../tables/", "tables/")
    open(os.path.join(tmp, "main.tex"), "w").write(src)
    for f in ("refs.tex", "neurips_2026.sty"):
        shutil.copy(os.path.join(HERE, "paper", f), os.path.join(tmp, f))
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=tmp, capture_output=True)
    return os.path.join(tmp, "main.pdf")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = build(tmp)
        log = open(os.path.join(tmp, "main.log"), errors="ignore").read()
        if not os.path.exists(pdf):
            print("BUILD FAILED"); print("\n".join(re.findall(r"^!.*", log, re.M)[:8])); return 2
        import fitz
        doc = fitz.open(pdf)
        refs = appx = None
        for i, page in enumerate(doc, 1):
            text = page.get_text()
            if refs is None and re.search(r"^\s*References\s*$", text, re.M):
                refs = i
            if appx is None and re.search(r"^\s*Appendix\s*$", text, re.M):
                appx = i
        if refs is None:
            print("could not find the References heading"); return 2
        content = refs - 1
        errs = len(re.findall(r"^!", log, re.M))
        undef = len(re.findall(r"undefined", log))
        print(f"  content pages   {content}/{LIMIT}   {'OK' if content <= LIMIT else 'OVER LIMIT'}")
        print(f"  references      page {refs}")
        if appx:
            print(f"  appendix        pages {appx}-{doc.page_count} "
                  f"({doc.page_count - appx + 1} pages, no limit)")
        print(f"  total           {doc.page_count} pages")
        print(f"  latex errors    {errs}    undefined refs {undef}")
        return 0 if content <= LIMIT and errs == 0 and undef == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
