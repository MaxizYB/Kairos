"""Render both published methods and every program, from archived client data."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kairos.audit import audit_results


def main():
    report=audit_results()
    programs={method:{r['program']:r['jct_s'] for r in
              json.loads((ROOT/'results'/method/'jobs.json').read_text())}
              for method in ('kairos','continuum')}
    with (ROOT/'results/program_jct.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['program','continuum_jct_s','kairos_jct_s','delta_s'])
        for p in sorted(programs['kairos']):
            c=programs['continuum'][p];k=programs['kairos'][p];writer.writerow([p,c,k,k-c])
    rows=['# Kairos 与 Continuum：已归档实验','',
        '单卡 RTX 4090、Qwen3-8B；16 programs、178 calls。每方法一次正向回放。','',
        '| 方法 | Mean JCT/s ↓ | P95 JCT/s ↓ | Wall/s ↓ |',
        '|---|---:|---:|---:|']
    for r in reversed(report['results']):
        label='Kairos' if r['method']=='kairos' else 'Continuum'
        rows.append(f"| {label} | {r['mean_jct_s']:.3f} | {r['p95_jct_s']:.3f} | {r['wall_s']:.3f} |")
    rows+=['','改善幅度：'+', '.join(f'{k}: {v:.2f}%' for k,v in report['reductions_percent'].items()),
        '','逐程序完整数据见 program_jct.csv；详细配置、指标口径与局限见 ../docs/EXPERIMENTS.md。',
        '这是归档测量结果，不是对重命名后代码新做的 GPU 验证。','']
    (ROOT/'results/README.md').write_text('\n'.join(rows))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
