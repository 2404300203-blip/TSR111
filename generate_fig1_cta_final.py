import csv
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OLD = Path(os.environ.get("FIG1_OLD_CSV", ROOT.parent / "fig1_updated_bubble_20260811" / "fig1_updated_bubble_source.csv"))
CTA = Path(os.environ.get("FIG1_CTA_CSV", ROOT / "evidence" / "deployment_export_20260812" / "cta_all_models" / "summary.csv"))
OUT = Path(os.environ.get("FIG1_OUTPUT_DIR", ROOT / "figures" / "fig1_cta_strict_deployment"))
OUT.mkdir(parents=True, exist_ok=True)

def main():
    with OLD.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with CTA.open(newline="", encoding="utf-8") as f:
        speed = {r["model"]: float(r["forward_fps_mean"]) for r in csv.DictReader(f)}
    speed.update({"SLANet":21.455258741974063, "DBM-SLANet":19.592180971390825})
    labels = {
        "DBM-SLANet": "DBM-SLANet (ours)",
        "Full Mamba, no presence": "Mamba, no presence",
        "Full Mamba, presence": "Mamba, presence",
        "Full Mamba, one-way": "Mamba, one-way",
        "Full DBM, d=64": "DBM d=64", "Full DBM, d_s=8": "DBM d_s=8",
        "Full DBM, d=192": "DBM d=192", "Full DBM, e=1": "DBM e=1",
        "Full Mamba, zero geometry": "Mamba, zero geometry",
        "True Horizontal-only": "Horizontal-only", "True Vertical-only": "Vertical-only",
    }
    for r in rows:
        name = r["model"]
        key = {"Cell Token only":"cell_token", "Lite Mamba":"lite_mamba"}.get(name, name)
        if r["family"] != "External":
            r["fps_cta500_strict"] = speed.get(key, speed.get(name, float(r["fps_cta500"])))
        else:
            r["fps_cta500_strict"] = float(r["fps_cta500"])
        r["label_final"] = labels.get(name, r["label"])
    with (OUT / "fig1_cta_strict_source.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model","label_final","family","status","teds_cta500","structure_accuracy_cta500","fps_cta500_strict","parameters","source_note"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k:r[k] for k in fields} for r in rows)
    colors = {"Baseline":"#4C78A8","Lightweight":"#E6A23C","Ablation":"#8395A7","Single-axis":"#3BA272","Proposed":"#D9534F","External":"#277B80"}
    params = [int(r["parameters"]) for r in rows]; pmin, pmax = math.log10(min(params)), math.log10(max(params))
    fig, ax = plt.subplots(figsize=(10.4,6.25), dpi=160); ax.set_facecolor("#FBFDFE")
    for r in rows:
        x=float(r["teds_cta500"]); y=float(r["fps_cta500_strict"]); acc=float(r["structure_accuracy_cta500"]); p=int(r["parameters"])
        area=100+1100*acc; norm=(math.log10(p)-pmin)/(pmax-pmin); area*=0.72+0.56*norm
        proposed=r["family"]=="Proposed"; ext=r["family"]=="External"
        ax.scatter(x,y,s=area,color=colors[r["family"]],alpha=.92,edgecolors="#A92F2B" if proposed else "white",linewidths=2 if proposed else 1,zorder=4)
        ax.annotate(r["label_final"],(x,y),xytext=(6,7),textcoords="offset points",fontsize=7.2,fontweight="bold" if proposed or ext else "normal",color="#A92F2B" if proposed else "#26343C",bbox=dict(facecolor="#FBFDFE",edgecolor="none",pad=.6,alpha=.88),arrowprops=dict(arrowstyle="-",lw=.45,color="#8395A7"))
    ax.set_xlim(.22,.75); ax.set_ylim(0,27); ax.set_xlabel("TEDS score on CTA-500 (n = 500)"); ax.set_ylabel("Strict synchronized inference speed on CTA-500 (FPS; higher = faster)")
    ax.grid(True,color="#D8E7EE",linewidth=.7); ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout()
    fig.savefig(OUT/"fig1_cta_strict_deployment.png",bbox_inches="tight"); fig.savefig(OUT/"fig1_cta_strict_deployment.pdf",bbox_inches="tight"); fig.savefig(OUT/"fig1_cta_strict_deployment.svg",bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__": main()
