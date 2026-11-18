"""仅负责显示真实运行结果；不改变任何窗口或点的位置。"""
import json
from pathlib import Path


def make_figures(examples, summary, diagnostic_summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Patch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC") if name in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.family": [chosen, "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    colors = {"固定网格": "#D38042", "四起点叠加": "#9276B7", "100米步长滑窗": "#5988BB", "精确滑窗": "#168A83"}
    ink, grey, competitor_color = "#233747", "#B4BDC6", "#AA577A"
    filenames = []
    sample = next(iter(examples.values()))
    block_count = sample["参数"]["宽度_米"] * sample["参数"]["高度_米"] // 1_000_000

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.18)
            filenames.append(str(path))
        plt.close(fig)

    def count(points, x, y):
        return sum(x <= px < x + 1000 and y <= py < y + 1000 for px, py, _ in points)

    def panel(ax, case, method, compact=False):
        points, result = case["点"], case["方法"][method]
        width, height = case["参数"]["宽度_米"], case["参数"]["高度_米"]
        color = colors[method]
        for x in range(0, width + 1, 1000):
            ax.axvline(x / 1000, lw=0.8, color="#D6DEE4", zorder=0)
        for y in range(0, height + 1, 1000):
            ax.axhline(y / 1000, lw=0.8, color="#D6DEE4", zorder=0)
        # 小数字显示原固定网格每格点数，便于人工复核。
        for x in range(0, width, 1000):
            for y in range(0, height, 1000):
                ax.text(x / 1000 + 0.06, y / 1000 + 0.86, str(count(points, x, y)),
                        fontsize=8, color="#8C9BA7", zorder=1)
        ax.scatter([p[0] / 1000 for p in points], [p[1] / 1000 for p in points],
                   s=30, color=grey, edgecolors="white", linewidths=0.45, zorder=3)
        x, y = result["x"], result["y"]
        members = [p for p in points if x <= p[0] < x + 1000 and y <= p[1] < y + 1000]
        assert len(members) == result["count"]
        ax.add_patch(Rectangle((x / 1000, y / 1000), 1, 1, facecolor=color, alpha=0.10, zorder=1))
        ax.add_patch(Rectangle((x / 1000, y / 1000), 1, 1, fill=False, edgecolor=color, lw=2.2, zorder=4))
        ax.scatter([p[0] / 1000 for p in members], [p[1] / 1000 for p in members],
                   s=37, color=color, edgecolors="white", linewidths=0.45, zorder=5)
        ax.scatter([(x + 500) / 1000], [(y + 500) / 1000], marker="+", s=90,
                   color="white", linewidths=4, zorder=6)
        ax.scatter([(x + 500) / 1000], [(y + 500) / 1000], marker="+", s=90,
                   color=ink, linewidths=1.6, zorder=7)
        competitor = case["竞争窗"]
        warning = competitor is not None and competitor["gap"] <= case["容差_点"]
        if method == "精确滑窗" and warning:
            cx, cy = competitor["x"], competitor["y"]
            assert count(points, cx, cy) == competitor["count"]
            ax.add_patch(Rectangle((cx / 1000, cy / 1000), 1, 1, fill=False,
                                   edgecolor=competitor_color, lw=1.7, linestyle=(0, (4, 3)), zorder=4))
        best = case["方法"]["精确滑窗"]["count"]
        ax.set_title(f"{method}｜主窗{result['count']}点｜缺口{best-result['count']}点",
                     loc="left", fontsize=11, fontweight="bold", color=color, pad=10)
        ax.set(xlim=(0, width / 1000), ylim=(0, height / 1000), xlabel="东向坐标（km）", ylabel="北向坐标（km）")
        ax.set_xticks(range(width // 1000 + 1)); ax.set_yticks(range(height // 1000 + 1))
        ax.set_aspect("equal", adjustable="box")
        if method == "精确滑窗" and not compact:
            message = (f"另有{competitor['count']}点竞争窗，差{competitor['gap']}点：保留不确定性"
                       if warning else "没有检测到容差内的不同点集竞争窗")
            ax.text(0.0, -0.19 if not compact else -0.21, message, transform=ax.transAxes,
                    fontsize=8.5, color=competitor_color if warning else "#667887")

    def legend(fig, y=0.012):
        fig.legend(handles=[Patch(facecolor=colors["固定网格"], label="橙色实框：固定网格主窗"),
                            Patch(facecolor=colors["精确滑窗"], label="青色实框：精确滑窗主窗"),
                            Line2D([], [], color=competitor_color, linestyle="--", label="紫色虚框：竞争提示窗，非第二个主窗"),
                            Line2D([], [], color=ink, marker="+", linestyle="none", label="＋：窗口中心；格角小数字：固定格点数")],
                   loc="lower center", bbox_to_anchor=(0.5, y), frameon=False, ncol=2, fontsize=9)

    main_cases = [examples["单热点_首次随机样本"], examples["双热点_首次随机样本"]]
    for index, case in enumerate(main_cases, 1):
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 7.0))
        fig.subplots_adjust(left=0.065, right=0.98, top=0.81, bottom=0.20, wspace=0.23)
        for ax, method in zip(axes, ("固定网格", "精确滑窗")):
            panel(ax, case, method)
        fig.suptitle(f"{case['场景']}｜全图{len(case['点'])}点，固定网格{block_count}块", x=0.065, y=0.97,
                     ha="left", fontsize=18, weight="bold", color=ink)
        fig.text(0.065, 0.90, f"首次随机样本，未按效果筛选｜种子{case['随机种子']}｜每种方法只选择一个1 km²主窗", color="#667887")
        legend(fig)
        save(fig, f"{index:02d}_{case['场景']}_首次随机地图")
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 12.6))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.89, bottom=0.105, wspace=0.23, hspace=0.48)
    for row, case in enumerate(main_cases):
        for col, method in enumerate(("固定网格", "精确滑窗")):
            panel(axes[row, col], case, method, compact=True)
    fig.suptitle(f"{block_count}格地图中的单热点与双热点", x=0.065, y=0.98, ha="left", fontsize=20, weight="bold", color=ink)
    fig.text(0.065, 0.935, "上排：单热点首次样本；下排：双热点首次样本。全部点均显示，未作局部裁切。", color="#667887")
    legend(fig, 0.005)
    save(fig, "00_单热点双热点对照总览")

    case = examples.get("双热点_跨区域竞争案例", main_cases[1])
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 7.0))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.81, bottom=0.20, wspace=0.23)
    for ax, method in zip(axes, ("固定网格", "精确滑窗")):
        panel(ax, case, method)
    fig.suptitle("选出一个主窗，同时报告跨区域竞争", x=0.065, y=0.97, ha="left", fontsize=18, weight="bold", color=ink)
    fig.text(0.065, 0.90, f"机制示例｜种子{case['随机种子']}｜{case['筛选说明']}", fontsize=9, color="#667887")
    legend(fig)
    save(fig, "03_双热点竞争_主窗与提示窗")

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 12.6))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.89, bottom=0.11, wspace=0.23, hspace=0.47)
    for ax, method in zip(axes.flat, colors):
        panel(ax, case, method, compact=True)
    fig.suptitle(f"同一张{block_count}格地图的四种搜索方法", x=0.065, y=0.98, ha="left", fontsize=20, weight="bold", color=ink)
    fig.text(0.065, 0.935, f"种子{case['随机种子']}｜方法只改变窗口起点的搜索范围，窗口面积均为1 km²。", color="#667887")
    legend(fig, 0.005)
    save(fig, "04_四种方法_同图对比")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    scenarios = ("单热点", "双热点")
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(colors):
            record = next(r for r in summary if r["场景"] == scenario and r["方法"] == method)
            position = i + (j - 1.5) * 0.18
            gap, hit = record["平均点数缺口"], record["最优值命中率"] * 100
            axes[0].bar(position, gap, width=0.17, color=colors[method], label=method if i==0 else None)
            axes[1].bar(position, hit, width=0.17, color=colors[method])
            axes[0].text(position, gap + 0.12, f"{gap:.2f}", ha="center", fontsize=8)
            axes[1].text(position, hit + 1.7, f"{hit:.1f}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks([0, 1], scenarios); ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)
    axes[0].set(ylabel="平均点数缺口（点）", ylim=(0, max(r['平均点数缺口'] for r in summary) + 1.5), title="固定边界损失能否减少")
    axes[1].set(ylabel="最优值命中率（%）", ylim=(0, 113), title="达到样本全局最大计数的比例")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.20, wspace=0.22)
    fig.suptitle(f"每类{summary[0]['地图数']}张随机地图的汇总结果", x=0.07, ha="left", fontsize=17, weight="bold", color=ink)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=4, frameon=False)
    fig.text(0.5, 0.01, "最优值命中率衡量当前样本的计数目标，不是真实热点识别准确率。", ha="center", fontsize=9, color="#667887")
    save(fig, "05_随机重复实验_方法比较")

    rules = ("朴素次优", "仅点集", "仅质心距离", "双条件")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5))
    for i, rule in enumerate(rules):
        single = next(r for r in diagnostic_summary if r["场景"] == "单热点" and r["规则"] == rule)
        dual = next(r for r in diagnostic_summary if r["场景"] == "双热点" and r["规则"] == rule)
        fp = single["假阳性率"]
        if fp is not None:
            axes[0].bar(i, fp * 100, color="#9EB9CC")
            axes[0].text(i, fp * 100 + 2, f"{fp:.0%}", ha="center", fontsize=9)
        for offset, key, color in ((-0.17, "精确率", "#559E94"), (0.17, "召回率", "#C69B69")):
            value = dual[key]
            if value is not None:
                axes[1].bar(i + offset, value * 100, width=0.30, color=color, label=key if i==0 else None)
                axes[1].text(i + offset, value * 100 + 2, f"{value:.0%}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(range(4), rules, rotation=15); ax.set_ylim(0, 115)
        ax.set_ylabel("比例（%）"); ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)
    axes[0].set_title("单热点：是否把邻近次优窗误当竞争")
    axes[1].set_title("双热点：与生成标签参照比较"); axes[1].legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.23, wspace=0.22)
    fig.suptitle("竞争诊断的规则消融", x=0.07, ha="left", fontsize=18, weight="bold", color=ink)
    fig.text(0.5, 0.055, "当前分离点群设定中，联合规则与单项结果相同，不能据此宣称联合规则更优。", ha="center", fontsize=9, color="#667887")
    save(fig, "06_竞争诊断_规则消融")
    (output / "图像清单.json").write_text(json.dumps(filenames, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存{len(filenames)//2}张图，各含PNG和PDF；每张地图只有一个实线主窗，虚线只表示竞争提示。")


if __name__ == "__main__":
    import sys
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("新版实验结果")
    examples = json.loads((directory / "可视化案例.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "实验概要.json").read_text(encoding="utf-8"))
    make_figures(examples, report["方法汇总"], report["诊断汇总"], directory / "图像")
