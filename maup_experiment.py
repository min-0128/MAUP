"""小样本MAUP实验：16格地图、单热点和双热点、单主窗选择及竞争诊断。

基础实验仅需Python 3.10以上标准库；--figures 另需Matplotlib。
例：python maup_experiment.py --runs 200 --figures --verify
"""
import argparse
from bisect import bisect_left
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from time import perf_counter

SIDE = 1000
SCENARIOS = ("单热点", "双热点")
METHODS = ("固定网格", "四起点叠加", "100米步长滑窗", "精确滑窗")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate(scenario, seed, cols=4, rows=4):
    """生成随机旋转椭圆内均匀点及全域背景；不预设哪个生成点群必胜。"""
    rng = random.Random(seed)
    width, height = cols * SIDE, rows * SIDE
    centers = [(rng.uniform(700, width - 700), rng.uniform(700, height - 700))]
    if scenario == "双热点":
        corners = [(x, y) for x in (700, width - 700) for y in (700, height - 700)]
        # 4×2地图中，正中央可能放不下相距1.6 km的第二点群；先保证可行性。
        while max(math.dist(centers[0], corner) for corner in corners) < 1700:
            centers[0] = (rng.uniform(700, width - 700), rng.uniform(700, height - 700))
        for _ in range(10000):
            candidate = (rng.uniform(700, width - 700), rng.uniform(700, height - 700))
            if math.dist(candidate, centers[0]) >= 1600:
                centers.append(candidate)
                break
        else:
            raise RuntimeError("无法在当前地图中放置两个分离点群。")
    n_a = rng.randint(12, 20)
    counts = [n_a] if scenario == "单热点" else [n_a, max(10, n_a - rng.randint(0, 3))]
    points, parameters = [], []
    for index, ((cx, cy), count) in enumerate(zip(centers, counts)):
        rx, ry, angle = rng.uniform(480, 680), rng.uniform(350, 600), rng.uniform(0, math.pi)
        label = "A" if index == 0 else "B"
        for _ in range(count):
            radius, theta = math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
            u, v = rx * radius * math.cos(theta), ry * radius * math.sin(theta)
            x = round(cx + u * math.cos(angle) - v * math.sin(angle))
            y = round(cy + u * math.sin(angle) + v * math.cos(angle))
            assert 0 <= x < width and 0 <= y < height
            points.append((x, y, label))
        parameters.append({"点群": label, "点数": count, "中心_米": [cx, cy],
                           "半轴_米": [rx, ry], "旋转角_弧度": angle})
    for _ in range(rng.randint(8, 12)):
        points.append((rng.randrange(width), rng.randrange(height), "背景"))
    return points, {"宽度_米": width, "高度_米": height, "点群参数": parameters}


class Axis:
    def __init__(self, coordinates, last):
        ordered = sorted(enumerate(coordinates), key=lambda item: item[1])
        self.values = [value for _, value in ordered]
        self.prefix = [0]
        for i, _ in ordered:
            self.prefix.append(self.prefix[-1] | (1 << i))
        self.starts = sorted({0, last} | {v for value in coordinates for v in (value, value - SIDE)
                                         if 0 <= v <= last})

    def mask(self, lo, hi):
        return self.prefix[bisect_left(self.values, hi)] ^ self.prefix[bisect_left(self.values, lo)]


class Geometry:
    """选择规则仅使用坐标与点编号；生成来源标签不进入选择或诊断。"""
    def __init__(self, points, width, height):
        self.points, self.width, self.height = points, width, height
        self.x = Axis([p[0] for p in points], width - SIDE)
        self.y = Axis([p[1] for p in points], height - SIDE)
        self.cache = {}

    def mask(self, x, y):
        return self.x.mask(x, x + SIDE) & self.y.mask(y, y + SIDE)

    def ids(self, mask):
        result = []
        while mask:
            bit = mask & -mask
            result.append(bit.bit_length() - 1)
            mask ^= bit
        return result

    def stats(self, mask):
        if mask not in self.cache:
            ids = self.ids(mask)
            mx = mean(self.points[i][0] for i in ids)
            my = mean(self.points[i][1] for i in ids)
            variance = mean((self.points[i][0] - mx) ** 2 + (self.points[i][1] - my) ** 2 for i in ids)
            self.cache[mask] = (mx, my, variance)
        return self.cache[mask]

    def rank(self, mask):
        mx, my, variance = self.stats(mask)
        return (-mask.bit_count(), variance, mx, my, mask)

    def representative(self, mask, origin, recenter=False):
        x, y = origin
        if recenter:
            # 在点集不变时改善位置表达；这一步不增加分数。
            ids = self.ids(mask)
            mx, my, _ = self.stats(mask)
            starts = []
            for axis, midpoint, domain in ((0, mx, self.width), (1, my, self.height)):
                values = [self.points[i][axis] for i in ids]
                lo, hi = max(0, max(values) - SIDE + 0.5), min(min(values), domain - SIDE)
                starts.append(min(max(midpoint - SIDE / 2, lo), hi))
            if self.mask(*starts) == mask:
                x, y = starts
        assert self.mask(x, y) == mask
        return {"x": x, "y": y, "mask": mask, "count": mask.bit_count()}


def search(geometry, method):
    start = perf_counter()
    if method == "精确滑窗":
        xs, ys = geometry.x.starts, geometry.y.starts
    else:
        step = {"固定网格": SIDE, "四起点叠加": SIDE // 2, "100米步长滑窗": 100}[method]
        xs = range(0, geometry.width - SIDE + 1, step)
        ys = range(0, geometry.height - SIDE + 1, step)
    memberships, checks = {}, 0
    for x in xs:
        xm = geometry.x.mask(x, x + SIDE)
        for y in ys:
            mask = xm & geometry.y.mask(y, y + SIDE)
            checks += 1
            if mask:
                memberships.setdefault(mask, (x, y))
    maximum = max(mask.bit_count() for mask in memberships)
    best_mask = min((mask for mask in memberships if mask.bit_count() == maximum), key=geometry.rank)
    result = geometry.representative(best_mask, memberships[best_mask], method == "精确滑窗")
    result.update(checks=checks, ms=1000 * (perf_counter() - start))
    return result, memberships


def dominant_source(geometry, mask):
    """仅用于仿真评价：至少60%的入选点来自同一生成点群才赋予A/B标签。"""
    counts = Counter(geometry.points[i][2] for i in geometry.ids(mask))
    for label in ("A", "B"):
        if 5 * counts[label] >= 3 * mask.bit_count():
            return label
    return "混合或背景"


def diagnose(geometry, memberships, primary, jaccard=0.2, separation=1.0, rule="仅点集"):
    primary_mask = primary["mask"]
    px, py, _ = geometry.stats(primary_mask)
    # 先按分数分层；只有某一分数层有合格竞争者时才在该层打破并列。
    levels = {}
    for mask in memberships:
        if mask != primary_mask:
            levels.setdefault(mask.bit_count(), []).append(mask)
    for count in sorted(levels, reverse=True):
        eligible = []
        for mask in levels[count]:
            overlap = (mask & primary_mask).bit_count() / (mask | primary_mask).bit_count()
            mx, my, _ = geometry.stats(mask)
            distance = math.hypot(mx - px, my - py) / SIDE
            good_j = overlap <= jaccard + 1e-12
            good_d = distance >= separation - 1e-12
            if rule == "朴素次优" or (rule == "仅点集" and good_j) or (rule == "仅质心距离" and good_d) or (rule == "双条件" and good_j and good_d):
                eligible.append(mask)
        if eligible:
            mask = min(eligible, key=geometry.rank)
            result = geometry.representative(mask, memberships[mask], True)
            mx, my, _ = geometry.stats(mask)
            result.update(gap=primary["count"] - count,
                          jaccard=(mask & primary_mask).bit_count() / (mask | primary_mask).bit_count(),
                          distance=math.hypot(mx - px, my - py) / SIDE)
            return result
    return None


def semantic_competition(geometry, memberships, primary, delta):
    """生成标签只定义评价参照；不为方法提供答案。混合主窗不纳入该指标。"""
    owner = dominant_source(geometry, primary["mask"])
    if owner not in ("A", "B"):
        return None
    other = "B" if owner == "A" else "A"
    return any(mask.bit_count() >= primary["count"] - delta
               and dominant_source(geometry, mask) == other for mask in memberships)


def safe_rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def independent_check(geometry, maximum):
    # 独立逐点计数，避免仅拿同一位集实现与自身作比较。
    best = max(sum(x <= px < x + SIDE and y <= py < y + SIDE for px, py, _ in geometry.points)
               for x in geometry.x.starts for y in geometry.y.starts)
    assert best == maximum, "位集搜索与独立逐点计数不一致。"


def experiment(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cols, rows = (4, 4) if args.blocks == 16 else (4, 2)
    point_rows, method_rows, diagnostic_rows, examples, map_rows = [], [], [], {}, []
    all_checks = 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial in range(args.runs):
            seed = args.seed + scenario_index * 1_000_003 + trial
            points, parameters = simulate(scenario, seed, cols, rows)
            geo = Geometry(points, cols * SIDE, rows * SIDE)
            results, exact = {}, None
            for method in METHODS:
                results[method], pool = search(geo, method)
                if method == "精确滑窗":
                    exact = pool
            main = results["精确滑窗"]
            assert all(result["count"] <= main["count"] for result in results.values())
            assert [results[m]["count"] for m in METHODS] == sorted(results[m]["count"] for m in METHODS), "嵌套搜索范围的最大点数应当单调不减。"
            if args.verify and trial < 3:
                independent_check(geo, main["count"])
                all_checks += 1
            default_competitor = None
            configurations = [(rule, args.delta, args.jaccard, args.separation)
                              for rule in ("朴素次优", "仅点集", "仅质心距离", "双条件")]
            configurations += [("双条件_容差0", 0, args.jaccard, args.separation),
                               ("双条件_容差2", 2, args.jaccard, args.separation),
                               ("双条件_J0", args.delta, 0.0, args.separation),
                               ("双条件_J0.5", args.delta, 0.5, args.separation),
                               ("双条件_距离0.75", args.delta, args.jaccard, 0.75),
                               ("双条件_距离1.25", args.delta, args.jaccard, 1.25)]
            for name, delta, j, separation in configurations:
                rule = name if name in ("朴素次优", "仅点集", "仅质心距离") else "双条件"
                competitor = diagnose(geo, exact, main, j, separation, rule)
                warning = competitor is not None and competitor["gap"] <= delta
                truth = semantic_competition(geo, exact, main, delta)
                if name == args.rule:
                    default_competitor = competitor
                diagnostic_rows.append({"场景": scenario, "重复编号": trial + 1, "随机种子": seed,
                    "规则": name, "容差_点": delta, "J阈值": j, "距离阈值_窗口边长": separation,
                    "主窗点数": main["count"], "竞争窗点数": None if competitor is None else competitor["count"],
                    "竞争差距_点": None if competitor is None else competitor["gap"], "发出竞争提示": int(warning),
                    "评价可用": int(truth is not None), "标签参照存在竞争": None if truth is None else int(truth),
                    "主窗来源": dominant_source(geo, main["mask"]),
                    "竞争窗来源": "无" if competitor is None else dominant_source(geo, competitor["mask"])})
            for method, result in results.items():
                union = (result["mask"] | main["mask"]).bit_count()
                method_rows.append({"场景": scenario, "重复编号": trial + 1, "随机种子": seed, "方法": method,
                    "总点数": len(points), "主窗点数": result["count"], "全局最大点数": main["count"],
                    "点数缺口": main["count"] - result["count"], "最优值命中": int(main["count"] == result["count"]),
                    "与精确主窗点集J": (result["mask"] & main["mask"]).bit_count() / union,
                    "左下角横坐标_米": result["x"], "左下角纵坐标_米": result["y"],
                    "实际窗口计数次数": result["checks"], "搜索耗时_毫秒": result["ms"]})
            for i, (x, y, source) in enumerate(points, 1):
                point_rows.append({"场景": scenario, "重复编号": trial + 1, "随机种子": seed,
                                   "点编号": i, "横坐标_米": x, "纵坐标_米": y, "生成来源": source})
            map_rows.append({"场景": scenario, "随机种子": seed, **parameters})
            snapshot = {"场景": scenario, "随机种子": seed, "重复编号": trial + 1, "点": points,
                        "参数": parameters, "方法": results, "竞争窗": default_competitor,
                        "容差_点": args.delta, "J阈值": args.jaccard, "距离阈值": args.separation,
                        "主诊断规则": args.rule}
            if trial == 0:
                examples[f"{scenario}_首次随机样本"] = {**snapshot, "筛选说明": "未筛选，使用第1次随机样本"}
            if results["固定网格"]["count"] < main["count"]:
                examples.setdefault(f"{scenario}_边界损失案例", {**snapshot, "筛选说明": "首次出现固定网格点数缺口大于0的样本，仅作机制示意"})
            if scenario == "双热点" and default_competitor and default_competitor["gap"] <= args.delta:
                owner = dominant_source(geo, main["mask"])
                other = dominant_source(geo, default_competitor["mask"])
                if owner in ("A", "B") and other in ("A", "B") and owner != other:
                    examples.setdefault("双热点_跨区域竞争案例", {**snapshot, "筛选说明": "首次出现不同生成点群且差距在容差内的样本，仅作机制示意"})
    summary = []
    for scenario in SCENARIOS:
        for method in METHODS:
            records = [r for r in method_rows if r["场景"] == scenario and r["方法"] == method]
            summary.append({"场景": scenario, "方法": method, "地图数": len(records),
                "平均点数缺口": mean(r["点数缺口"] for r in records),
                "最优值命中率": mean(r["最优值命中"] for r in records),
                "平均窗口计数次数": mean(r["实际窗口计数次数"] for r in records),
                "搜索耗时中位数_毫秒": median(r["搜索耗时_毫秒"] for r in records)})
    diagnostic_summary = []
    for scenario in SCENARIOS:
        for name in dict.fromkeys(r["规则"] for r in diagnostic_rows):
            records = [r for r in diagnostic_rows if r["场景"] == scenario and r["规则"] == name]
            valid = [r for r in records if r["评价可用"]]
            tp = sum(r["发出竞争提示"] and r["标签参照存在竞争"] for r in valid)
            fp = sum(r["发出竞争提示"] and not r["标签参照存在竞争"] for r in valid)
            fn = sum(not r["发出竞争提示"] and r["标签参照存在竞争"] for r in valid)
            tn = len(valid) - tp - fp - fn
            diagnostic_summary.append({"场景": scenario, "规则": name, "地图数": len(records), "评价可用数": len(valid),
                "提示次数": sum(r["发出竞争提示"] for r in records), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "精确率": safe_rate(tp, tp + fp), "召回率": safe_rate(tp, tp + fn), "假阳性率": safe_rate(fp, fp + tn)})
    write_csv(output / "仿真点数据.csv", point_rows)
    write_csv(output / "逐地图方法结果.csv", method_rows)
    write_csv(output / "方法汇总.csv", summary)
    write_csv(output / "竞争诊断明细.csv", diagnostic_rows)
    write_csv(output / "竞争诊断汇总.csv", diagnostic_summary)
    (output / "生成参数.json").write_text(json.dumps(map_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "可视化案例.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"设置": vars(args), "方法汇总": summary, "诊断汇总": diagnostic_summary,
              "独立逐点核验地图数": all_checks, "点记录数": len(point_rows)}
    (output / "实验概要.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in summary:
        print(f"{record['场景']}｜{record['方法']}：平均缺口{record['平均点数缺口']:.2f}点，最优值命中率{record['最优值命中率']:.1%}")
    print(f"已保存{args.runs * 2}张随机地图的数值结果；固定网格为{cols}×{rows}，每个方法只选择一个主窗。")
    if args.figures:
        from plot_results import make_figures
        make_figures(examples, summary, diagnostic_summary, output / "图像")
    return report


class ChineseParser(argparse.ArgumentParser):
    def format_help(self):
        return super().format_help().replace("usage:", "用法：", 1)

    def format_usage(self):
        return super().format_usage().replace("usage:", "用法：", 1)

    def error(self, message):
        self.print_usage()
        self.exit(2, "参数错误：" + message.replace("unrecognized arguments:", "无法识别的参数：") + "\n")


def main():
    parser = ChineseParser(description=__doc__, add_help=False)
    parser._positionals.title = "位置参数"
    parser._optionals.title = "可选参数"
    parser.add_argument("-h", "--help", action="help", help="显示中文帮助")
    parser.add_argument("--runs", type=int, default=200, help="每类场景的地图数，默认200")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--blocks", type=int, choices=(8, 16), default=16, help="固定网格块数；8块为4×2，16块为4×4")
    parser.add_argument("--delta", type=int, default=1, help="竞争容差，单位为点数")
    parser.add_argument("--jaccard", type=float, default=0.2)
    parser.add_argument("--separation", type=float, default=1.0, help="点集质心距离阈值，以窗口边长为单位")
    parser.add_argument("--rule", choices=("仅点集", "双条件"), default="仅点集", help="默认用较简点集规则；双条件额外要求质心距离分离")
    parser.add_argument("--output", default="新版实验结果")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--verify", action="store_true", help="每类前3张地图额外使用独立逐点计数核验")
    args = parser.parse_args()
    if args.runs < 1 or args.delta < 0 or not 0 <= args.jaccard <= 1 or args.separation < 0:
        parser.error("次数须大于0；容差和距离阈值非负；J阈值介于0与1。")
    if args.figures:
        try:
            import matplotlib.pyplot
        except ImportError:
            parser.error("绘图需要Matplotlib，请运行 python -m pip install matplotlib。")
    experiment(args)


if __name__ == "__main__":
    main()
