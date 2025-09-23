import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import csv

# Usage examples:
# python plot_cmc.py --json cmc.json --out cmc.png --subset a
# python plot_cmc.py --json cmc.json --out cmc_with_paper.png --subset a \
#        --extra_csv paper_curve.csv --extra_label "Liu14 (paper)"
# CSV format: rank,score (1-based rank index) or rank1,rank2,... on header

def load_cmc(json_path: Path, subset: str) -> Dict[str, List[float]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    scores = data["scores"][subset]
    return scores  # mapping: method -> [score@rank1, ..., score@rankK]


def load_extra_csv(csv_path: Path) -> List[float]:
    # Support two formats:
    # 1) rows with rank,score
    # 2) single row with columns rank1,rank2,...
    arr: List[float] = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return arr
        # Try header-style first
        header = [c.strip().lower() for c in rows[0]]
        if any(h.startswith('rank') for h in header) and len(rows) >= 2:
            # assume row 1 is scores or we parse columns with rank*
            for h, v in zip(header, rows[1]):
                if h.startswith('rank'):
                    try:
                        arr.append(float(v))
                    except ValueError:
                        pass
            return arr
        # else assume two-column rank,score rows
        for r in rows:
            if len(r) < 2:
                continue
            try:
                rank = int(float(r[0]))
                score = float(r[1])
            except ValueError:
                continue
            # ensure list length
            while len(arr) < rank:
                arr.append(arr[-1] if arr else 0.0)
            arr[rank-1] = score
    return arr


def ensure_monotonic(arr: List[float]) -> List[float]:
    if not arr:
        return arr
    out = []
    m = 0.0
    for v in arr:
        m = max(m, float(v))
        out.append(min(1.0, m))
    return out


def plot_cmc(curves: Dict[str, List[float]], out_path: Path, title: str, ylim: Optional[List[float]] = None):
    plt.figure(figsize=(7,5), dpi=160)
    # style cycle
    styles = [
        ('#1f77b4','-'),
        ('#ff7f0e','--'),
        ('#2ca02c','-.'),
        ('#d62728',':'),
        ('#9467bd','-'),
        ('#8c564b','--'),
    ]
    ranks = None
    for idx, (label, ys) in enumerate(curves.items()):
        xs = np.arange(1, len(ys)+1)
        ranks = xs
        color, ls = styles[idx % len(styles)]
        plt.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=2)
        # mark rank-1
        plt.scatter([1], [ys[0]], color=color, s=24, zorder=5)
    plt.xlabel('Rank k')
    plt.ylabel('Identification Rate (CMC)')
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(loc='lower right')
    if ylim:
        plt.ylim(ylim)
    if ranks is not None:
        plt.xlim(1, ranks[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='Path to cmc.json')
    ap.add_argument('--subset', default='a', help='Which subset to plot (e.g., a,b,c,d)')
    ap.add_argument('--out', default='cmc.png', help='Output image path')
    ap.add_argument('--title', default='CMC Curves', help='Plot title')
    ap.add_argument('--extra_csv', default=None, help='Optional CSV of additional curve (rank,score)')
    ap.add_argument('--extra_label', default='Paper (Liu14)', help='Label for the extra curve')
    ap.add_argument('--approx_from', default=None, help='Approximate the extra curve by scaling from an existing method name')
    ap.add_argument('--approx_rank1', type=float, default=None, help='Target rank-1 for approximation (e.g., 0.268)')
    args = ap.parse_args()

    json_path = Path(args.json)
    out_path = Path(args.out)

    scores = load_cmc(json_path, args.subset)
    curves: Dict[str, List[float]] = {}
    # Use a stable order
    order = [m for m in ["Proposed","Feng et al.","STFT","Gabor"] if m in scores]
    for m in order:
        curves[m] = list(map(float, scores[m]))

    # Add extra curve if provided
    extra: Optional[List[float]] = None
    if args.extra_csv:
        extra = load_extra_csv(Path(args.extra_csv))
    elif args.approx_from and args.approx_rank1 is not None:
        base = curves.get(args.approx_from)
        if base:
            base = list(base)
            if base[0] <= 0:
                scale = 1.0
            else:
                scale = float(args.approx_rank1) / float(base[0])
            extra = [min(1.0, max(0.0, v*scale)) for v in base]
            extra = ensure_monotonic(extra)
    if extra:
        curves[args.extra_label] = extra

    title = args.title or f"CMC ({args.subset})"
    plot_cmc(curves, out_path, title, ylim=[0.0, 1.0])


if __name__ == '__main__':
    main()
