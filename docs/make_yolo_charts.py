"""README용 YOLO 모델 선정 그래프를 docs/images/ 에 만든다.

실행: python3 docs/make_yolo_charts.py  (yolo_compare/runs 가 있어야 한다)

웹캠(n급 6종)과 AMR(YOLOv8 n~x) 수치는 src/mini_vision/models/yolo선택과정_*.md 의 표,
학습 곡선과 sweep 은 yolo_compare/runs 의 results.csv / results.jsonl 에서 읽는다.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = WS / 'docs' / 'images'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': ['Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumSquare', 'DejaVu Sans'],
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#b9b8b1',
    'axes.labelcolor': '#52514e',
    'xtick.color': '#52514e',
    'ytick.color': '#52514e',
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlecolor': '#0b0b0b',
    'figure.facecolor': '#fcfcfb',
    'axes.facecolor': '#fcfcfb',
    'savefig.facecolor': '#fcfcfb',
    'grid.color': '#e6e5df',
    'grid.linewidth': 0.8,
})

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
PICK = SERIES[0]
OTHER = '#c9c8c0'
TEXT = '#0b0b0b'
MUTED = '#52514e'


def hbars(ax, labels, values, picked, title, unit, fmt='{:.1f}'):
    colors = [PICK if l == picked else OTHER for l in labels]
    y = range(len(labels))
    ax.barh(y, values, color=colors, height=0.62, edgecolor='#fcfcfb', linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_title(title, loc='left')
    ax.set_xlabel(unit)
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    top = max(values) if max(values) > 0 else 1
    ax.set_xlim(0, top * 1.22)
    for i, v in enumerate(values):
        ax.text(v + top * 0.02, i, fmt.format(v), va='center', fontsize=9,
                color=TEXT, fontweight='bold' if labels[i] == picked else 'normal')
    for t in ax.get_yticklabels():
        if t.get_text() == picked:
            t.set_fontweight('bold')
            t.set_color(TEXT)


# ------------------------------------------------------------------ 1. 웹캠
web = ['YOLOv8n', 'YOLOv9t', 'YOLOv10n', 'YOLO11n', 'YOLO12n', 'YOLO26n']
web_fp = [0, 1, 1, 0, 0, 1]             # conf 0.8, valid+test 68장
web_cpu = [22.9, 28.4, 22.7, 24.1, 30.2, 22.2]
web_gpu = [3.8, 8.4, 4.5, 5.0, 6.5, 5.2]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
hbars(axes[0], web, web_fp, 'YOLOv8n', '① conf 0.8 오탐 수 (68장)', '개', '{:.0f}')
hbars(axes[1], web, web_cpu, 'YOLOv8n', '② CPU 추론 시간', 'ms / 이미지')
hbars(axes[2], web, web_gpu, 'YOLOv8n', '③ GPU 추론 시간', 'ms / 이미지')
axes[0].set_xlim(0, 1.6)
axes[0].set_xticks([0, 1])
fig.suptitle('고정 웹캠 모델: n급 6종 비교 (mAP50 모두 0.995, Recall 모두 1.000)',
             x=0.01, ha='left', fontsize=12, fontweight='bold', color=TEXT)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / 'yolo_webcam_selection.png', dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 2. AMR
amr = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
err_025 = [1, 1, 1, 3, 0]               # FP + FN, valid+test 54개 정답
err_08 = [5, 0, 0, 17, 2]
amr_cpu = [33.9, 97.6, 248.6, 501.1, 787.5]
amr_gpu = [3.3, 4.6, 12.6, 19.3, 31.7]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
ax = axes[0]
y = list(range(len(amr)))
h = 0.36
ax.barh([i - h / 2 for i in y], err_025, height=h, color=SERIES[1], label='conf 0.25',
        edgecolor='#fcfcfb', linewidth=2)
ax.barh([i + h / 2 for i in y], err_08, height=h, color=SERIES[0], label='conf 0.8',
        edgecolor='#fcfcfb', linewidth=2)
for i in y:
    ax.text(err_025[i] + 0.3, i - h / 2, str(err_025[i]), va='center', fontsize=8.5, color=TEXT)
    ax.text(err_08[i] + 0.3, i + h / 2, str(err_08[i]), va='center', fontsize=8.5, color=TEXT,
            fontweight='bold' if amr[i] == 'YOLOv8s' else 'normal')
ax.set_yticks(y, amr)
ax.invert_yaxis()
ax.set_xlim(0, 20)
ax.set_title('① 오류 수 (오탐+미탐, 정답 54개)', loc='left')
ax.set_xlabel('개')
ax.grid(axis='x')
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='lower right', fontsize=9)
for t in ax.get_yticklabels():
    if t.get_text() == 'YOLOv8s':
        t.set_fontweight('bold')
hbars(axes[1], amr, amr_cpu, 'YOLOv8s', '② CPU 추론 시간', 'ms / 이미지', '{:.0f}')
axes[1].axvline(100, color='#e34948', lw=1.2, ls='--')
axes[1].text(112, 0.55, 'CPU 10Hz 한계 100ms', color='#a33232', fontsize=8.5, va='center')
hbars(axes[2], amr, amr_gpu, 'YOLOv8s', '③ GPU 추론 시간', 'ms / 이미지')
fig.suptitle('AMR 카메라 모델: YOLOv8 크기 5종 비교 (mAP50 모두 0.995)',
             x=0.01, ha='left', fontsize=12, fontweight='bold', color=TEXT)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / 'yolo_amr_selection.png', dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 3. AMR n급 학습 곡선
runs = WS / 'yolo_compare' / 'runs'
curve_models = [('yolov8n', 'YOLOv8n'), ('yolov9t', 'YOLOv9t'), ('yolov10n', 'YOLOv10n'),
                ('yolo11n', 'YOLO11n'), ('yolo12n', 'YOLO12n'), ('yolo26n', 'YOLO26n')]
fig, ax = plt.subplots(figsize=(9, 4.2))
for (folder, label), color in zip(curve_models, SERIES):
    with open(runs / folder / 'results.csv') as f:
        rows = list(csv.DictReader(f))
    ep = [int(float(r['epoch'])) for r in rows]
    m = [float(r['metrics/mAP50-95(B)']) for r in rows]
    ax.plot(ep, m, color=color, lw=2, label=label)
ax.set_ylim(0.5, 0.95)
ax.set_xlabel('epoch')
ax.set_ylabel('val mAP50-95')
ax.grid(axis='y')
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc='lower right', fontsize=9)
ax.set_title('추가 검증 ①: AMR 데이터셋(amr-v1)에서 n급 6종 학습 곡선', loc='left')
fig.tight_layout()
fig.savefig(OUT / 'yolo_amr_ncls_curves.png', dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 4. YOLO26 sweep
groups = defaultdict(list)
with open(runs / 'sweep' / 'results.jsonl') as f:
    for line in f:
        r = json.loads(line)
        groups[(r['Size'], r['Optimizer'])].append(r)
sizes = ['n', 's', 'm']
opts = ['AdamW', 'SGD', 'MuSGD']
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for key, ax, title, unit in [
    ('Test mAP50-95', axes[0], '① test mAP50-95 (seed 3개)', 'test mAP50-95'),
    ('Inference(ms/img)', axes[1], '② GPU 추론 시간 (seed 3개)', 'ms / 이미지'),
]:
    for j, (opt, color) in enumerate(zip(opts, SERIES)):
        for i, size in enumerate(sizes):
            vals = [r[key] for r in groups[(size, opt)]]
            x = i + (j - 1) * 0.22
            ax.scatter([x] * len(vals), vals, s=22, color=color, alpha=0.45,
                       edgecolor='#fcfcfb', linewidth=1, zorder=2)
            ax.errorbar(x, mean(vals), yerr=stdev(vals), fmt='o', ms=8, color=color,
                        mec='#fcfcfb', mew=2, capsize=4, lw=2, zorder=3,
                        label=opt if i == 0 else None)
    ax.set_xticks(range(3), ['YOLO26n', 'YOLO26s', 'YOLO26m'])
    ax.set_title(title, loc='left')
    ax.set_ylabel(unit)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc='upper right', ncol=3, fontsize=9, title='optimizer', title_fontsize=9)
fig.suptitle('추가 검증 ②: YOLO26 크기 × optimizer × seed 3개 (큰 점 = 평균 ± 표준편차)',
             x=0.01, ha='left', fontsize=12, fontweight='bold', color=TEXT)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / 'yolo26_sweep.png', dpi=150)
plt.close(fig)

print('saved:', sorted(p.name for p in OUT.glob('*.png')))
