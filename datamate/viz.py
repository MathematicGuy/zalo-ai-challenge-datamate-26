"""Notebook display helpers (ported from Zalo-Track-2): Mermaid statecharts rendered to PNG, a question's path, Input/Output boxes."""

import base64
import hashlib
import html
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from IPython.display import HTML, Markdown, display

# bgColor takes a hex code without "#"; a bare "white" is ignored and gives a transparent background
MERMAID_INK = "https://mermaid.ink/img/{}?type=png&bgColor={}"

# rendered PNGs are cached by source hash under runs/viz_cache (gitignored): a rerun needs no network
CACHE_DIR = Path(__file__).resolve().parent.parent / "runs" / "viz_cache"
CACHE_VERSION = "1"  # bump when the theme changes to drop the old cache

_session = requests.Session()  # one connection for every render

# every chart is rendered twice (light/dark) and the notebook shows the one matching its theme
THEMES = {
    "light": {
        "bg": "FFFFFF",
        "vars": {
            "background": "#ffffff", "primaryColor": "#e8f0fe", "primaryBorderColor": "#4a6fd1",
            "primaryTextColor": "#1f2937", "mainBkg": "#e8f0fe", "textColor": "#1f2937",
            "lineColor": "#374151", "transitionColor": "#374151", "transitionLabelColor": "#1f2937",
            "labelBackgroundColor": "#ffffff", "edgeLabelBackground": "#ffffff",
            "stateLabelColor": "#1f2937", "stateBkg": "#e8f0fe", "specialStateColor": "#374151",
        },
    },
    "dark": {
        "bg": "1E1E2E",
        "vars": {
            "background": "#1e1e2e", "primaryColor": "#2f3b5c", "primaryBorderColor": "#8ab4f8",
            "primaryTextColor": "#f1f5f9", "mainBkg": "#2f3b5c", "textColor": "#f1f5f9",
            "lineColor": "#e2e8f0", "transitionColor": "#e2e8f0", "transitionLabelColor": "#f1f5f9",
            "labelBackgroundColor": "#1e1e2e", "edgeLabelBackground": "#1e1e2e",
            "stateLabelColor": "#f1f5f9", "stateBkg": "#2f3b5c", "specialStateColor": "#e2e8f0",
        },
    },
}

# VS Code puts vscode-dark / vscode-high-contrast on the output <body>; JupyterLab uses data-jp-theme-light
_DARK = 'body.vscode-dark, body.vscode-high-contrast:not(.vscode-high-contrast-light), body[data-jp-theme-light="false"]'
_CHART_CSS = (
    "<style>"
    + ", ".join(f"{s} .dm-light" for s in _DARK.split(", ")) + "{display:none !important}"
    + ", ".join(f"{s} .dm-dark" for s in _DARK.split(", ")) + "{display:block !important}"
    + "</style>"
)


def _themed(src: str, mode: str) -> str:
    """Insert the theme init into the Mermaid source (after a leading --- block if any)."""
    init = "%%{init: " + json.dumps({"theme": "base", "themeVariables": THEMES[mode]["vars"]}) + "}%%"
    if src.startswith("---"):
        end = src.index("\n---", 3) + 4
        return f"{src[:end]}\n{init}{src[end:]}"
    return f"{init}\n{src}"


def _render_png(src: str, bg: str) -> bytes:
    """PNG of Mermaid source `src`: from the disk cache, else from mermaid.ink (then cached)."""
    key = hashlib.sha256(f"{CACHE_VERSION}\n{bg}\n{src}".encode("utf-8")).hexdigest()
    path = CACHE_DIR / f"{key}.png"
    try:
        return path.read_bytes()
    except OSError:
        pass
    encoded = base64.urlsafe_b64encode(src.encode("utf-8")).decode("ascii")
    resp = _session.get(MERMAID_INK.format(encoded, bg), timeout=20)
    resp.raise_for_status()
    try:  # atomic write; a disk error does not break the display
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
    except OSError:
        pass
    return resp.content


def _show_variants(build_src, width: int | None = None) -> None:
    """Render `build_src(mode)` in both themes and show the one matching the notebook. Offline: print the Mermaid source instead."""
    sources = {mode: _themed(build_src(mode), mode) for mode in THEMES}
    with ThreadPoolExecutor(max_workers=len(THEMES)) as pool:
        futures = {mode: pool.submit(_render_png, src, THEMES[mode]["bg"]) for mode, src in sources.items()}
    pngs, error = {}, None
    for mode, fut in futures.items():
        try:
            pngs[mode] = fut.result()
        except requests.RequestException as e:
            error = e
    if not pngs:
        display(Markdown(f"_(could not render the chart: {error}. Mermaid source:)_\n```mermaid\n{build_src('light').strip()}\n```"))
        return
    size = f"width:{width}px;" if width else ""
    imgs = []
    for mode, png in pngs.items():
        # only one variant rendered: show it always, without the theme classes
        cls = f' class="dm-{mode}"' if len(pngs) == len(THEMES) else ""
        hidden = "display:none;" if mode == "dark" and len(pngs) == len(THEMES) else "display:block;"
        imgs.append(
            f'<img{cls} alt="chart" src="data:image/png;base64,{base64.b64encode(png).decode("ascii")}" '
            f'style="{hidden}{size}max-width:100%;border-radius:8px">'
        )
    display(HTML(_CHART_CSS + "".join(imgs)))


def show_statechart(mermaid_src: str, width: int | None = None) -> None:
    """Render Mermaid (e.g. stateDiagram-v2 or flowchart) as an embedded PNG that follows the light/dark theme."""
    src = mermaid_src.strip()
    _show_variants(lambda mode: src, width)


TRACE_STYLES = {
    "light": {
        "pre": "fill:#e8f0fe,stroke:#4a6fd1,color:#1f2937",
        "llm": "fill:#fff4d6,stroke:#c99a00,color:#1f2937",
        "exec": "fill:#e6f4ea,stroke:#2e7d32,color:#1f2937",
        "error": "fill:#fde2e1,stroke:#c62828,color:#1f2937",
        "post": "fill:#f3e8fd,stroke:#7b3fbf,color:#1f2937",
    },
    "dark": {
        "pre": "fill:#1f3358,stroke:#8ab4f8,color:#f1f5f9",
        "llm": "fill:#4a3b0a,stroke:#f2c94c,color:#f1f5f9",
        "exec": "fill:#16391f,stroke:#6fcf7f,color:#f1f5f9",
        "error": "fill:#4a1d1d,stroke:#ff8a80,color:#f1f5f9",
        "post": "fill:#3b2459,stroke:#c79bf2,color:#f1f5f9",
    },
}


def _mermaid_label(text: str, limit: int = 60) -> str:
    # characters that break Mermaid labels are replaced, long labels cut
    text = str(text).replace('"', "'").replace("{", "(").replace("}", ")").replace(";", ",").replace("#", "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def show_trace(trace: list[dict], title: str = "") -> None:
    """The actual path of one question through the pipeline: steps `{"kind": "pre|llm|exec|error|post", "label", "detail"}`
    in time order become states; error steps are red."""
    def build(mode: str) -> str:
        rows = ["stateDiagram-v2"]
        if title:
            rows.insert(0, f'---\ntitle: "{_mermaid_label(title, 90).replace(":", " -")}"\n---')
        for kind, style in TRACE_STYLES[mode].items():
            rows.append(f"    classDef {kind} {style}")
        prev = "[*]"
        for i, step in enumerate(trace):
            label = _mermaid_label(step["label"])
            if step.get("detail"):
                label += "<br/>" + _mermaid_label(step["detail"])
            rows.append(f'    state "{label}" as s{i}')
            rows.append(f"    {prev} --> s{i}")
            rows.append(f"    class s{i} {step['kind']}")
            prev = f"s{i}"
        rows.append(f"    {prev} --> [*]")
        return "\n".join(rows)

    _show_variants(build)


def _to_html(obj) -> str:
    if isinstance(obj, pd.DataFrame):
        return obj.head(8).to_html(index=False, border=0)
    if isinstance(obj, pd.Series):
        return obj.head(8).to_frame().to_html(border=0)
    if isinstance(obj, (dict, list)):
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    else:
        text = str(obj)
    if len(text) > 3000:
        text = text[:3000] + "\n… (truncated)"
    return f"<pre style='white-space:pre-wrap;margin:0'>{html.escape(text)}</pre>"


def show_io(inp, out, title: str = "") -> None:
    """A step's Input and Output side by side."""
    box = "flex:1;min-width:280px;border:1px solid #ccc;border-radius:6px;padding:8px;overflow:auto"
    display(HTML(
        f"<div><b>{html.escape(title)}</b></div>"
        f"<div style='display:flex;gap:12px;flex-wrap:wrap'>"
        f"<div style='{box}'><b>📥 Input</b>{_to_html(inp)}</div>"
        f"<div style='{box}'><b>📤 Output</b>{_to_html(out)}</div>"
        f"</div>"
    ))
