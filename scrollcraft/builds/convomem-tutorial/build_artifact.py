"""Build artifact.html: a single self-contained copy of index.html for publishing.

The artifact host wraps the page in its own document and only allows Google Fonts as an
external stylesheet, so the engine CSS, the engine JS and data.js are inlined here and the
page's own <html>/<head>/<body> wrapper is removed.

    python3 scrollcraft/builds/convomem-tutorial/build_artifact.py
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE = "ConvoMem Rollout Inspector"


def script(src: str) -> str:
    return "<script>\n" + src.replace("</", "<\\/") + "\n</script>"


def main() -> None:
    page = (HERE / "index.html").read_text()
    css = (HERE / "scrollcraft.css").read_text()
    head = page.split("<head>", 1)[1].split("</head>", 1)[0]
    body = page.split("<body>", 1)[1].split("</body>", 1)[0]

    head = re.sub(r"<meta charset[^>]*>\s*", "", head)
    head = re.sub(r"<meta name=\"viewport\"[^>]*>\s*", "", head)
    head = re.sub(r"<title>.*?</title>", f"<title>{TITLE}</title>", head)
    head = re.sub(r'<link rel="icon" href="[^"]*">\s*', "", head)
    head = head.replace('<link rel="stylesheet" href="scrollcraft.css">', "<style>\n" + css + "\n</style>")

    body = body.replace('<script src="data.js"></script>', script((HERE / "data.js").read_text()))
    body = body.replace('<script src="scrollcraft.js"></script>', script((HERE / "scrollcraft.js").read_text()))
    out = head.strip() + "\n\n" + body.strip() + "\n"
    leftover = re.findall(r'<(?:script|link)[^>]+(?:src|href)="(?!https://fonts\.)[^"#]+"', out)
    assert not leftover, leftover
    (HERE / "artifact.html").write_text(out)
    print(f"artifact.html {len(out) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
