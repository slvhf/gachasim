import numpy as np
from numba import jit, prange
import time
import math
import os

# ####################################
# 用户可修改参数 (独立运行时使用)
# ####################################
TARGET_TYPE = 'pulls'
TARGET_COUNT = 100
MAX_PULLS = 6000
SIMULATIONS = 1_000_000
SPECIFIC_PULLS = [260, 320, 940]
PERCENTILES = [10, 25, 50, 75, 90]
INTERVAL_WIDTH = 1
SHOW_DISTRIBUTION = 1
BATCH_SIZE = 5_000

# ####################################


@jit(nopython=True, parallel=True, fastmath=True)
def cpu_simulate_batch(target_mode, target_count, max_pulls, batch_size):
    results = np.zeros((batch_size, 6), dtype=np.int64)
    
    for i in prange(batch_size):
        k, gold, up, pulls_used = 0, 0, 0, 0
        guaranteed_up, achieved = False, False
        last_gold, last_up = 0, 0
        ming_guang_counter = 1
        is_fixed_mode = (target_mode == 2)
        
        while True:
            if is_fixed_mode and pulls_used >= max_pulls: break
            if not is_fixed_mode and (achieved or pulls_used >= max_pulls): break
            
            k += 1
            threshold = 600
            if k >= 74:
                added = 6000 * (k - 73)
                threshold = min(600 + added, 100000)
            
            if np.random.randint(0, 100000) < threshold:
                new_pulls = pulls_used + k
                if is_fixed_mode and new_pulls > max_pulls: break
                
                pulls_used, gold, last_gold = new_pulls, gold + 1, new_pulls
                is_up = False
                
                if guaranteed_up:
                    is_up, guaranteed_up = True, False
                else:
                    if ming_guang_counter >= 3:
                        is_up, ming_guang_counter = True, 1
                    else:
                        if np.random.randint(0, 1000) < 500:
                            is_up = True
                            ming_guang_counter = max(0, ming_guang_counter - 1)
                        else:
                            guaranteed_up = True
                            ming_guang_counter = min(3, ming_guang_counter + 1)
                
                if is_up:
                    up, last_up = up + 1, pulls_used
                k = 0
            
            if is_fixed_mode and pulls_used + k > max_pulls: break
            if not is_fixed_mode and ((target_mode == 0 and gold >= target_count) or (target_mode == 1 and up >= target_count)):
                achieved = True
                break
        
        # --- 修正: 将复杂的切片赋值拆分为简单的单元素赋值 ---
        results[i, 0] = np.int64(pulls_used)
        results[i, 1] = np.int64(1 if achieved else 0)
        results[i, 2] = np.int64(gold)
        results[i, 3] = np.int64(up)
        results[i, 4] = np.int64(last_gold)
        results[i, 5] = np.int64(last_up)
        
    return results

def calculate_distribution(data, max_value, interval):
    if len(data) == 0: return np.array([]), np.array([])
    max_val_data = data.max() if len(data) > 0 else 0
    max_value = max(max_value, max_val_data)
    bins = np.arange(0, max_value + interval + 1, interval)
    hist, _ = np.histogram(data, bins=bins)
    return hist, bins

def print_distribution(hist, bins, title, unit, show_interval=True):
    interval_str = f"（{INTERVAL_WIDTH}{unit}）" if show_interval else ""
    print(f"\n📊 {title}分布{interval_str}")
    print("区间范围       | 数量        | 占比 (%)   | 累计 (%)")
    print("------------------------------------------------------")
    total = hist.sum()
    if total == 0: print("⚠ 无数据"); return
    cumulative = 0.0
    for i in range(len(hist)):
        if hist[i] == 0: continue
        start, end = bins[i], bins[i+1] - 1
        percentage = hist[i] / total * 100
        cumulative += percentage
        range_str = f"{int(start):4}-{int(end):4}" if start != end else f"{int(start):4}  "
        print(f"{range_str} | {hist[i]:10,} | {percentage:8.6f}% | {cumulative:8.6f}%")

def calculate_statistics(data, total_achieved, max_pulls, percentiles, specific_pulls):
    stats = {'percentiles': {}, 'specific_pulls': {}, 'min': 0, 'max': 0}
    if total_achieved == 0 or len(data) == 0: return stats
    sorted_data = np.sort(data)
    stats['min'], stats['max'] = sorted_data[0], sorted_data[-1]
    cumulative = np.arange(1, len(sorted_data)+1) / len(sorted_data) * 100
    for p in percentiles:
        index = min(max(np.searchsorted(cumulative, p), 0), len(sorted_data)-1)
        stats['percentiles'][p] = sorted_data[index]
    for pull in specific_pulls:
        stats['specific_pulls'][pull] = (np.sum(sorted_data <= pull) / len(sorted_data)) * 100
    return stats

def save_results_to_file(filename, params, results_data):
    try:
        output_dir = "simdata"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        full_path = os.path.join(output_dir, filename)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n" + "Gacha Simulation Results\n" + "="*50 + "\n")
            f.write(f"Target Type: {params['target_type']}\n")
            if params['target_type'] == 'pulls':
                f.write(f"Max Pulls per Sim: {int(params['max_pulls']):,}\n")
            else:
                f.write(f"Target Count: {int(params['target_count']):,}\n")
            f.write(f"Total Simulations: {int(params['simulations']):,}\n\n")

            for dist_name, data in results_data.items():
                f.write("-" * 50 + f"\nDistribution for: {dist_name}\n" + "-" * 50 + "\n")
                f.write(f"{'Count':<10} | {'Simulations':<15} | {'Percentage (%)'}\n" + "-" * 50 + "\n")
                if len(data) == 0: f.write("No data available.\n\n"); continue
                counts, total_valid_sims = np.bincount(data), len(data)
                for value, num_sims in enumerate(counts):
                    if num_sims > 0:
                        percentage = (num_sims / total_valid_sims) * 100
                        f.write(f"{value:<10} | {int(num_sims):<15,} | {percentage:.8f}%\n")
                f.write("\n")
        print(f"\n💾 详细结果已保存到文件: {full_path}")
    except Exception as e:
        print(f"\n❌ 保存文件时出错: ", e)

def run_simulation(target_type, target_count, max_pulls, simulations, batch_size, 
                   show_distribution=True, specific_pulls=None, percentiles=None):
    
    target_count = int(target_count)
    max_pulls = int(max_pulls)
    simulations = int(simulations)
    batch_size = int(batch_size)
    specific_pulls = specific_pulls if specific_pulls is not None else []
    percentiles = percentiles if percentiles is not None else []

    if target_type.lower() == 'pulls':
        target_mode = 2; actual_max_pulls = max_pulls
    elif target_type.lower() == 'gold':
        target_mode = 0; actual_max_pulls = target_count * 100
    elif target_type.lower() == 'up':
        target_mode = 1; actual_max_pulls = target_count * 200
    else:
        raise ValueError("无效目标类型，必须是'gold'/'up'/'pulls'")

    if target_mode == 2:
        filename_part = f"{target_type.lower()}_{max_pulls}"
    else:
        filename_part = f"{target_type.lower()}_{target_count}"
    filename = f"gachasim_{filename_part}_{simulations}.txt"

    print(f"▶ 开始CPU模拟：{simulations:,}次 ", end='')
    if target_mode == 2: print(f"固定抽数模式：{max_pulls}抽/次")
    else: print(f"目标{target_count}个{target_type}（最大抽数限制：{actual_max_pulls}）")
    
    all_results, start_time = [], time.time()
    global_stats = {'total_pulls': 0, 'total_gold_pulls': 0, 'total_up_pulls': 0, 
                    'total_gold': 0, 'total_up': 0, 'total_consumed': 0}
    
    remaining = simulations
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batch_results = cpu_simulate_batch(target_mode, target_count, actual_max_pulls, current_batch)
        all_results.append(batch_results)
        
        global_stats['total_pulls'] += batch_results[:, 0].sum()
        global_stats['total_gold'] += batch_results[:, 2].sum()
        global_stats['total_up'] += batch_results[:, 3].sum()
        global_stats['total_gold_pulls'] += batch_results[:, 4].sum()
        global_stats['total_up_pulls'] += batch_results[:, 5].sum()
        
        if target_mode == 2:
            global_stats['total_consumed'] += max_pulls * current_batch
        
        remaining -= current_batch
        print(f"▷ 进度 {100*(simulations-remaining)/simulations:.2f}% | 耗时 {time.time() - start_time:.1f}s", end='\r')
    
    if target_mode != 2:
        global_stats['total_consumed'] = global_stats['total_pulls']

    full_results = np.concatenate(all_results)
    
    up_rate = 0.0
    if global_stats['total_gold'] > 0:
        up_rate = min(global_stats['total_up'] / global_stats['total_gold'] * 100, 100.0)
    
    actual_avg_gold = global_stats['total_gold_pulls'] / global_stats['total_gold'] if global_stats['total_gold'] > 0 else 0.0
    combined_avg_gold = global_stats['total_consumed'] / global_stats['total_gold'] if global_stats['total_gold'] > 0 else 0.0
    actual_avg_up = global_stats['total_up_pulls'] / global_stats['total_up'] if global_stats['total_up'] > 0 else 0.0
    combined_avg_up = global_stats['total_consumed'] / global_stats['total_up'] if global_stats['total_up'] > 0 else 0.0
    
    print("\n\n⭐ 全局统计")
    print(f"- 总抽数: {global_stats.get('total_pulls', 0):,}抽")
    print(f"- 实际抽金消耗: {global_stats.get('total_gold_pulls', 0):,}抽（最后一次出金总和）")
    print(f"- 实际抽UP消耗: {global_stats.get('total_up_pulls', 0):,}抽（最后一次UP总和）")
    print(f"- 综合消耗: {global_stats.get('total_consumed', 0):,}抽（{'固定抽数×次数' if target_mode==2 else '与实际消耗相同'}）")
    print(f"- 总出金次数: {global_stats.get('total_gold', 0):,} (UP率: {up_rate:.6f}%)")
    print(f"- 实际平均抽金消耗: {actual_avg_gold:.6f} 抽/金（真实抽数/出金数）")
    print(f"- 综合平均抽金消耗: {combined_avg_gold:.6f} 抽/金（理论抽数/出金数）")
    print(f"- UP实际平均消耗: {actual_avg_up:.6f} 抽/UP（真实抽数/UP数）")
    print(f"- UP综合平均消耗: {combined_avg_up:.6f} 抽/UP（理论抽数/UP数）")

    success_count = 0
    if target_mode != 2:
        success_mask = full_results[:, 1] == 1
        success_count = full_results[:, 1].sum()
        print(f"\n⭐ 目标达成统计")
        print(f"- 成功率: {success_count / simulations * 100:.6f}%")
        if success_count > 0:
            success_pulls = full_results[success_mask, 0]
            print(f"- 平均消耗抽数: {success_pulls.mean():.6f}抽（成功案例）")
            pulls_hist, pulls_bins = calculate_distribution(success_pulls, actual_max_pulls, INTERVAL_WIDTH)
            if show_distribution:
                print_distribution(pulls_hist, pulls_bins, "消耗抽数", "抽")
            stats = calculate_statistics(success_pulls, success_count, actual_max_pulls, percentiles, specific_pulls)
            print("\n📈 关键百分位统计")
            for p, val in stats['percentiles'].items(): print(f"{p}% | {val:8}抽 | 有{p}%的模拟消耗≤此抽数")
    else:
        golds_data, ups_data, non_ups_data = full_results[:, 2], full_results[:, 3], full_results[:, 2] - full_results[:, 3]
        print("\n⭐ 固定抽数模式统计")
        print(f"- 平均出金数: {golds_data.mean():.6f}个/次")
        print(f"- 平均UP数: {ups_data.mean():.6f}个/次")
        print(f"- 平均非UP数: {non_ups_data.mean():.6f}个/次")
        if show_distribution:
            max_dist_val = max(10, int(max_pulls / 60)) if max_pulls > 0 else 10
            gold_hist, gold_bins = calculate_distribution(golds_data, max_dist_val, 1)
            up_hist, up_bins = calculate_distribution(ups_data, max_dist_val, 1)
            non_up_hist, non_up_bins = calculate_distribution(non_ups_data, max_dist_val, 1)
            print_distribution(gold_hist, gold_bins, "出金数量", "个", False)
            print_distribution(up_hist, up_bins, "UP数量", "个", False)
            print_distribution(non_up_hist, non_up_bins, "非UP数量", "个", False)

    params_to_save = {'target_type': target_type, 'max_pulls': max_pulls, 'target_count': target_count, 'simulations': simulations}
    results_to_save = {}
    if target_mode == 2:
        results_to_save['Gold'], results_to_save['UP'], results_to_save['Non-UP'] = full_results[:, 2], full_results[:, 3], full_results[:, 2] - full_results[:, 3]
    elif success_count > 0:
        results_to_save['Pulls to achieve target'] = full_results[full_results[:, 1] == 1, 0]
    
    if results_to_save:
        save_results_to_file(filename, params_to_save, results_to_save)

    global_stats['up_rate_percent'] = up_rate
    global_stats['actual_avg_gold'] = actual_avg_gold
    global_stats['combined_avg_gold'] = combined_avg_gold
    global_stats['actual_avg_up'] = actual_avg_up
    global_stats['combined_avg_up'] = combined_avg_up
    return global_stats

if __name__ == "__main__":
    print("--- 脚本独立运行 ---")
    run_simulation(
        target_type=TARGET_TYPE,
        target_count=TARGET_COUNT,
        max_pulls=MAX_PULLS,
        simulations=SIMULATIONS,
        batch_size=BATCH_SIZE,
        show_distribution=SHOW_DISTRIBUTION,
        specific_pulls=SPECIFIC_PULLS,
        percentiles=PERCENTILES
    )
    print("\n--- 模拟完成 ---")