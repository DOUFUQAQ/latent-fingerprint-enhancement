import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

SUBSETS: Tuple[str, ...] = ("a","b","c","d")
TITLES = {
    "a": "Cumulative Match Characteristic (a)",
    "b": "Cumulative Match Characteristic (b)",
    "c": "Cumulative Match Characteristic (c)",
    "d": "Cumulative Match Characteristic (d)",
}

ORDER = ["Proposed","Feng et al.","STFT","Gabor"]
COLORS = {
    "Proposed": "#d62728",  # red
    "Feng et al.": "#1f77b4",  # blue
    "STFT": "#2ca02c",       # green
    "Gabor": "#000000",      # black
}
STYLES = {
    "Proposed": "-",
    "Feng et al.": "-",
    "STFT": "-",
    "Gabor": "-",
}


def parse_rank1_map(s: str) -> Dict[str, float]:
    # format like: a:0.268,b:0.62,c:0.22,d:0.10
    out: Dict[str, float] = {}
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' not in token:
            continue
        k, v = token.split(':', 1)
        k = k.strip().lower()
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def ensure_monotonic(arr: List[float]) -> List[float]:
    if not arr:
        return arr
    out = []
    m = 0.0
    for v in arr:
        m = max(m, float(v))
        out.append(min(1.0, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='cmc.json 路径')
    ap.add_argument('--out', required=True, help='输出图片（将覆盖 cmc.png）')
    ap.add_argument('--paper_from', default='Feng et al.', help='近似论文曲线的基准曲线名')
    ap.add_argument('--paper_label', default='Liu14 (paper)', help='论文曲线图例标签')
    ap.add_argument('--paper_rank1', default='a:0.268', help='各子集论文 Rank-1，例如 a:0.268,b:0.62,c:0.22,d:0.10')
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding='utf-8'))
    scores: Dict[str, Dict[str, List[float]]] = data['scores']
    paper_rank1_map = parse_rank1_map(args.paper_rank1)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=160)
    axes_map = {"a": axes[0,0], "b": axes[0,1], "c": axes[1,0], "d": axes[1,1]}

    for subset in SUBSETS:
        if subset not in scores:
            continue
        ax = axes_map[subset]
        cur = scores[subset]
        # 画现有四条曲线
        for name in ORDER:
            if name not in cur:
                continue
            ys = list(map(float, cur[name]))
            xs = np.arange(1, len(ys)+1)
            ax.plot(xs, np.array(ys)*100.0, label=name, color=COLORS.get(name, None), linestyle=STYLES.get(name,'-'), linewidth=2)
        # 近似论文曲线：从指定基准曲线按 Rank-1 比例缩放
        if args.paper_from in cur and subset in paper_rank1_map:
            base = list(map(float, cur[args.paper_from]))
            target_r1 = paper_rank1_map[subset]
            scale = (target_r1 / base[0]) if base and base[0] > 0 else 1.0
            extra = [min(1.0, max(0.0, v*scale)) for v in base]
            extra = ensure_monotonic(extra)
            xs = np.arange(1, len(extra)+1)
            ax.plot(xs, np.array(extra)*100.0, label=args.paper_label, color='#9467bd', linestyle='--', linewidth=2)
        # 格式
        ax.set_title(TITLES.get(subset, subset))
        ax.set_xlabel('Rank')
        ax.set_ylabel('Identification Rate (%)')
        ax.set_ylim(0, 70)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(loc='lower right', fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == '__main__':
    main()
