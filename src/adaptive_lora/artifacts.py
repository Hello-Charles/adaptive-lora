"""M4 的无额外依赖结果输出：独占 JSON/CSV、单轴 SVG 和可离线交互的 HTML。

这里只呈现已完成校准的均值，不执行模型、重算概率或训练。输出目录由调用者新建，
历史文件绝不覆盖；目录含校准样本相关信息，发布前仍由调用者审查隐私与许可证。
"""

import csv
from html import escape
import json
import math
from pathlib import Path

from adaptive_lora.selection import rank_layers, select_top_k


def write_json(path, value):
    """先完整序列化有限 JSON，再独占创建文件，拒绝覆盖和 NaN/Infinity 伪 JSON。"""
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(text)


def _plot_data(result, selected):
    """图的下标必须与真实层号相同；不偷偷重排或让 Top-K 标记变成另一条数值系列。"""
    rows = result["layer_importance"]
    scores = [row["importance"] for row in rows]
    rank_layers(scores)
    if [row["layer_index"] for row in rows] != list(range(len(rows))) or result["layer_count"] != len(rows):
        raise ValueError("Layer rows must be complete and in zero-based index order")
    selected = list(selected)
    if any(type(index) is not int or not 0 <= index < len(rows) for index in selected) or len(set(selected)) != len(selected):
        raise ValueError("Selected layers must be unique legal zero-based indices")
    return rows, scores, set(selected)


# 色值来自 dataviz 参考 palette 的 slot 1；明/暗表面分别验证，不按大小/排名变色。
# 单系列无颜色图例；Top-K 用层号后的 * 和表格文字标识，避免只靠颜色传达选层。
_SVG_STYLE = """
svg { --surface:#fcfcfb; --ink:#0b0b0b; --secondary:#52514e; --grid:#e1e0d9;
      --axis:#c3c2b7; --series:#2a78d6; --hatch-ink:#1c5cab; font:13px 'Segoe UI','Microsoft YaHei',sans-serif; }
svg[data-theme="dark"] { --surface:#1a1a19; --ink:#f0efec; --secondary:#c3c2b7;
      --grid:#2c2c2a; --axis:#383835; --series:#3987e5; --hatch-ink:#184f95; }
text { fill:var(--ink); } .secondary,.tick { fill:var(--secondary); }
.tick { font-size:12px; font-variant-numeric:tabular-nums; } .grid { stroke:var(--grid); stroke-width:1; }
.bar { fill:var(--series); } .hit { fill:transparent; stroke:transparent; }
.mark:focus { outline:none; } .mark:hover .bar,.mark:focus .bar { filter:brightness(1.12); }
.mark:focus .hit { stroke:var(--ink); } .textured .bar { fill:url(#hatch); }
@media print, (forced-colors:active) { .bar { fill:url(#hatch); } }
"""


def render_importance_svg(result, selected_layers):
    """按层序画单系列柱图：零基线、最大 24px 柱宽、4px 数据端圆角及完整命中区域。

    长度唯一编码 I_l，* 只表示被选中；最大值附一个数字，其他值通过每柱 title/焦点提示
    和 HTML 表格读取。SVG 的颜色是固定角色变量，HTML 可显式切换明暗/纹理，不依赖 CDN。
    """
    rows, scores, selected = _plot_data(result, selected_layers)
    width, height = 1000, 520
    left, right, top, baseline = 70, 24, 112, 422
    plot_width, plot_height = width - left - right, baseline - top
    maximum = max(scores)
    # 留出标注余量，选 1/2/5 的整洁 tick；全零数据仍使用合法的零起始轴。
    raw_step = (maximum * 1.15 if maximum else 1.0) / 4
    power = 10 ** math.floor(math.log10(raw_step))
    step = next(factor * power for factor in (1, 2, 5, 10) if factor * power >= raw_step)
    ticks = math.ceil((maximum * 1.15 if maximum else 1.0) / step)
    ceiling = ticks * step
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"'
        ' data-theme="light" data-baseline="0" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Frozen-base layer importance</title>',
        f'<desc id="chart-desc">Mean absolute probability changes over {result["n"]} inputs. '
        f'One independent Attention and MLP zero-ablation per layer. Asterisk marks Top-{len(selected)} selection.</desc>',
        f'<style>{_SVG_STYLE}</style>',
        '<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse">'
        '<rect width="8" height="8" fill="var(--series)"/>'
        '<path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="var(--hatch-ink)" stroke-width="1.5"/>'
        '</pattern></defs>',
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>',
        '<text x="28" y="32" font-size="23" font-weight="600">Frozen-base layer importance</text>',
        f'<text x="28" y="58" class="secondary">N = {result["n"]} · {len(rows)} layers · independent Attention / MLP ablations</text>',
        '<text x="28" y="88" class="secondary">I_l · mean absolute probability change (not accuracy)</text>',
    ]
    for tick in range(ticks + 1):
        value = tick * step
        y = baseline - value / ceiling * plot_height
        parts.extend([
            f'<line x1="{left}" x2="{width - right}" y1="{y:.3f}" y2="{y:.3f}" class="grid"/>',
            f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" class="tick">{value:.4g}</text>',
        ])
    slot = plot_width / len(rows)
    bar_width = min(24, slot - 4)
    if bar_width <= 0:
        raise ValueError("Too many layers for this fixed-size chart; use a table or a larger plot")
    for index, value in enumerate(scores):
        x = left + index * slot + (slot - bar_width) / 2
        y = baseline - value / ceiling * plot_height
        radius = min(4, max(0, baseline - y) / 2, bar_width / 2)
        tooltip = f"Layer {index} | I_l = {value:.9g} | {'selected' if index in selected else 'not selected'}"
        # 路径只让柱的上端圆角，底部始终锚在零基线；透明槽覆盖小值/零值也可 hover/键盘聚焦。
        path = (f"M{x:.3f},{baseline} V{y + radius:.3f} Q{x:.3f},{y:.3f} {x + radius:.3f},{y:.3f} "
                f"H{x + bar_width - radius:.3f} Q{x + bar_width:.3f},{y:.3f} {x + bar_width:.3f},{y + radius:.3f} "
                f"V{baseline} Z")
        parts.extend([
            f'<g class="mark" tabindex="0" role="img" aria-label="{escape(tooltip, quote=True)}" data-tooltip="{escape(tooltip, quote=True)}">',
            f'<title>{escape(tooltip)}</title><path class="bar" d="{path}"/>',
            f'<rect class="hit" x="{left + index * slot:.3f}" y="{top}" width="{slot:.3f}" height="{plot_height}"/>',
            '</g>',
            f'<text x="{x + bar_width / 2:.3f}" y="447" text-anchor="middle" class="tick">{index}{"*" if index in selected else ""}</text>',
        ])
        if maximum > 0 and index == scores.index(maximum):
            parts.append(f'<text x="{x + bar_width / 2:.3f}" y="{y - 10:.3f}" text-anchor="middle">{value:.4g}</text>')
    parts.extend([
        f'<text x="{width / 2}" y="478" text-anchor="middle" class="secondary">Layer index (zero-based)</text>',
        f'<text x="28" y="505" class="secondary">* Selected Top-{len(selected)} · all values in the accompanying CSV / HTML table</text>',
        '</svg>',
    ])
    return "\n".join(parts)


_HTML_STYLE = """
body { color-scheme:light; --surface:#fcfcfb; --ink:#0b0b0b; --secondary:#52514e; --rule:#e1e0d9;
       margin:24px; background:var(--surface); color:var(--ink); font:15px 'Segoe UI','Microsoft YaHei',sans-serif; }
body[data-theme="dark"] { color-scheme:dark; --surface:#1a1a19; --ink:#f0efec; --secondary:#c3c2b7; --rule:#2c2c2a; }
main { max-width:1100px; margin:auto; } h1 { font-size:24px; } p { color:var(--secondary); line-height:1.6; }
a { color:var(--ink); } .controls { display:flex; gap:16px; align-items:center; margin:16px 0; }
button { padding:6px 12px; background:var(--surface); color:var(--ink); border:1px solid var(--rule); border-radius:4px; }
.chart { overflow-x:auto; } svg { display:block; width:100%; min-width:880px; height:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; margin-top:16px; }
th,td { text-align:right; padding:9px 12px; border-bottom:1px solid var(--rule); } th:first-child { text-align:left; }
caption { text-align:left; padding:12px; font-weight:600; } #tooltip { position:fixed; z-index:2; max-width:380px;
          padding:10px; background:var(--surface); color:var(--ink); border:1px solid var(--rule); border-radius:4px; pointer-events:none; }
@media print { .controls,#tooltip { display:none; } body { margin:0; } }
"""

_HTML_SCRIPT = """
// 仅展示本地既有数值，不联网/重算分数；tooltip 用 textContent，避免把标签当 HTML 执行。
const tip = document.getElementById('tooltip');
const svg = document.querySelector('svg');
function showTip(event) {
  tip.textContent = event.currentTarget.dataset.tooltip;
  tip.hidden = false;
  const box = event.currentTarget.getBoundingClientRect();
  const x = event.clientX || box.left + 12;
  const y = event.clientY || box.top + 12;
  tip.style.left = Math.max(8, Math.min(x + 14, innerWidth - tip.offsetWidth - 8)) + 'px';
  tip.style.top = Math.max(8, Math.min(y + 14, innerHeight - tip.offsetHeight - 8)) + 'px';
}
document.querySelectorAll('.mark').forEach(mark => {
  mark.addEventListener('pointermove', showTip);
  mark.addEventListener('focusin', showTip);
  mark.addEventListener('pointerleave', () => { tip.hidden = true; });
  mark.addEventListener('focusout', () => { tip.hidden = true; });
});
// 明暗各有选定 palette，不对颜色自动反相；纹理仅由读者选择或 print/forced-colors 启用。
document.getElementById('theme-toggle').addEventListener('click', event => {
  const dark = document.body.dataset.theme !== 'dark';
  document.body.dataset.theme = svg.dataset.theme = dark ? 'dark' : 'light';
  event.currentTarget.textContent = dark ? 'Switch to light' : 'Switch to dark';
});
document.getElementById('texture-toggle').addEventListener('change', event => {
  svg.classList.toggle('textured', event.currentTarget.checked);
});
"""


def render_importance_html(result, selected_layers):
    """自包含离线图/表，原始数值不由 tooltip 独占，支持键盘、显式主题和纹理。

    只把已校验的整数层号/有限数值插入页面；无外部字体、CDN、采集或网络请求。
    表格提供两个分量，图只编码合计 I_l，不引入第二个数值轴或伪装为下游准确率。
    """
    rows, _scores, selected = _plot_data(result, selected_layers)
    table = "\n".join(
        f'<tr><th scope="row">{row["layer_index"]}</th><td>{row["importance"]:.9g}</td>'
        f'<td>{row["attention_mean_abs_delta"]:.9g}</td><td>{row["mlp_mean_abs_delta"]:.9g}</td>'
        f'<td>{"Selected" if row["layer_index"] in selected else "—"}</td></tr>' for row in rows
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive-LoRA calibration importance</title><style>{_HTML_STYLE}</style></head>
<body data-theme="light"><main><h1>Adaptive-LoRA · calibration evidence</h1>
<p>Frozen-base zero-ablation only. No adapter training or downstream accuracy claim.
<a href="resolved_config.json">Configuration</a> · <a href="layer_importance.csv">Raw CSV</a> ·
<a href="selected_layers.json">Ranked Top-K</a></p>
<div class="controls"><button id="theme-toggle" type="button">Switch to dark</button>
<label><input id="texture-toggle" type="checkbox"> Accessible texture</label></div>
<div class="chart">{render_importance_svg(result, selected_layers)}</div>
<p>I_l = mean_x (|q − q_l^A| + |q − q_l^MLP|). Every probability measures the same original t*.
Layer indices start at 0. Hover or focus a bar for values; the table below does not require interaction.</p>
<table><caption>Full layer importance · N = {result["n"]}</caption><thead><tr>
<th scope="col">Layer</th><th scope="col">I_l</th><th scope="col">Attention mean |Δq|</th>
<th scope="col">MLP mean |Δq|</th><th scope="col">Selection</th></tr></thead><tbody>{table}</tbody></table>
</main><div id="tooltip" role="status" hidden></div><script>{_HTML_SCRIPT}</script></body></html>
'''


def write_calibration_artifacts(directory, result, *, top_k):
    """将已完成的核心结果导出到调用者的新目录，返回供 CLI 记录的全层排名/Top-K。

    所有目标文件先预检，实际仍用 x 独占创建防竞争覆盖；写入中断时可能留有部分文件，
    成功标记须由调用者在全部输出完成后另写，不能把“存在 CSV”视为完整运行成功。
    """
    directory = Path(directory)
    scores = [row["importance"] for row in result["layer_importance"]]
    order = rank_layers(scores)
    selected = select_top_k(scores, top_k)
    _plot_data(result, selected)
    names = ("layer_importance.json", "selected_layers.json", "layer_importance.csv", "importance.svg", "importance.html")
    if any((directory / name).exists() or (directory / name).is_symlink() for name in names):
        raise FileExistsError("Calibration artifacts already exist; use a new output directory")
    ranks = {index: rank + 1 for rank, index in enumerate(order)}
    rows = [{**row, "rank": ranks[row["layer_index"]], "selected": row["layer_index"] in selected}
            for row in result["layer_importance"]]
    selection = {
        "index_base": 0, "mode": "accuracy: Top-K over all layers", "top_k": top_k, "layer_count": len(scores),
        "selected_layer_indices": selected, "selected_layer_indices_ascending": sorted(selected),
        "ranked_layer_indices": order, "tie_break": "descending importance, then ascending layer index [复现决策]",
    }
    write_json(directory / "layer_importance.json", {"n": result["n"], "index_base": 0, "layers": rows})
    write_json(directory / "selected_layers.json", selection)
    with (directory / "layer_importance.csv").open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("layer_index", "importance", "attention_mean_abs_delta", "mlp_mean_abs_delta", "rank", "selected"))
        writer.writeheader()
        writer.writerows(rows)
    for name, text in (("importance.svg", render_importance_svg(result, selected)), ("importance.html", render_importance_html(result, selected))):
        with (directory / name).open("x", encoding="utf-8") as output:
            output.write(text)
    return selection
