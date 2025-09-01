# 导入必要的库
import numpy as np
from numba import cuda, NumbaPerformanceWarning # Numba 的 CUDA 模块
from numba.cuda.random import create_xoroshiro128p_states # GPU 随机数生成器
import time
import math
import warnings
import os

# 忽略 Numba 在某些情况下可能发出的性能警告，保持输出整洁
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# ####################################
# 用户可修改参数 (当此脚本独立运行时使用)
# ####################################

# --- 模拟模式配置 ---
TARGET_TYPE = 'pulls'
TARGET_COUNT = 1
MAX_PULLS = 20000 

# --- 性能与精度配置 ---
SIMULATIONS = 1_000_000
BATCH_SIZE = 100_000     # 批处理大小 (仅用于 'gold'/'up' 模式的进度反馈)

# --- 'pulls' 模式专属配置 ---
PULLS_RATIO = np.array([4, 1, 4, 1])

# --- 统计输出配置 ---
SPECIFIC_PULLS = [260, 320, 940]
PERCENTILES = [10, 25, 50, 75, 90]
INTERVAL_WIDTH = 1
SHOW_DISTRIBUTION = 1 # 设为 1 以显示所有分布统计

# ####################################
# 核心模拟常量 (通常不需要修改)
# ####################################

# --- CUDA 全局配置 ---
THREADS_PER_BLOCK = 1024  # 每个 CUDA 块包含的线程数
RAND_SEED = 42            # 随机数种子

# --- 'pulls' 模式卡池配置 ---
ERA_POOL_SIZES = np.array([5, 6, 7, 8])
SPECIAL_ITEM_THRESHOLD = 7
MAX_UNIQUE_NON_UP_CHARS = 15 # 为收藏柜数组分配空间


@cuda.jit(device=True)
def generate_rand_float(rng_states, tid):
    """
    CUDA 设备函数：为当前线程生成一个 [0.0, 1.0) 之间的随机浮点数。
    """
    return cuda.random.xoroshiro128p_uniform_float32(rng_states, tid)


@cuda.jit
def cuda_gacha_kernel(results, max_pulls, target_mode, target_count, total_simulations, rng_states):
    """
    为 'gold' 和 'up' 模式设计的 CUDA 核函数。
    """
    tid = cuda.grid(1)
    if tid >= total_simulations:
        return
    
    k, gold, up, pulls_used, ming_guang_counter = 0, 0, 0, 0, 1
    guaranteed_up, achieved, last_gold, last_up = False, False, 0, 0

    while not (achieved or pulls_used >= max_pulls):
        k += 1
        threshold = 0.006
        if k >= 74:
            threshold = min(0.006 + 0.06 * (k - 73), 1.0)
        
        if generate_rand_float(rng_states, tid) <= threshold:
            pulls_used += k
            gold += 1
            last_gold = pulls_used
            is_up = False
            
            if guaranteed_up:
                is_up, guaranteed_up = True, False
            else:
                if ming_guang_counter >= 3:
                    is_up, ming_guang_counter = True, 1
                else:
                    if generate_rand_float(rng_states, tid) <= 0.5:
                        is_up, ming_guang_counter = True, max(0, ming_guang_counter - 1)
                    else:
                        guaranteed_up, ming_guang_counter = True, min(3, ming_guang_counter + 1)
            
            if is_up:
                up += 1
                last_up = pulls_used
            k = 0

        if (target_mode == 0 and gold >= target_count) or (target_mode == 1 and up >= target_count):
            achieved = True

    results[tid, 0] = np.int64(pulls_used)
    results[tid, 1] = np.int64(1 if achieved else 0)
    results[tid, 2] = np.int64(gold)
    results[tid, 3] = np.int64(up)
    results[tid, 4] = np.int64(last_gold)
    results[tid, 5] = np.int64(last_up)


@cuda.jit
def cuda_pulls_mode_kernel(final_results, final_collections, max_pulls, era_boundaries, pool_sizes, item_threshold, simulations, rng_states):
    """
    为 'pulls' 模式设计的 CUDA 核函数。
    """
    tid = cuda.grid(1)
    if tid >= simulations:
        return
    
    k, gold, up, non_up, special_items = 0, 0, 0, 0, 0
    guaranteed_up = False
    ming_guang_counter = 1
    
    collection = cuda.local.array(MAX_UNIQUE_NON_UP_CHARS, dtype=np.int32)
    for i in range(MAX_UNIQUE_NON_UP_CHARS):
        collection[i] = 0
    
    for pull in range(1, max_pulls + 1):
        era_idx = 0
        for i in range(len(era_boundaries) - 1):
            if pull > era_boundaries[i]:
                era_idx = i + 1
        current_pool_size = pool_sizes[era_idx]

        k += 1
        threshold = 0.006
        if k >= 74:
            threshold = min(0.006 + 0.06 * (k - 73), 1.0)

        if generate_rand_float(rng_states, tid) <= threshold:
            is_up = False
            gold += 1
            
            if guaranteed_up:
                is_up, guaranteed_up = True, False
            else:
                if ming_guang_counter >= 300: 
                    is_up, ming_guang_counter = True, 1
                else:
                    if generate_rand_float(rng_states, tid) <= 0.5:
                        is_up, ming_guang_counter = True, max(0, ming_guang_counter - 1)
                    else:
                        guaranteed_up, ming_guang_counter = True, min(3, ming_guang_counter + 1)
            
            if is_up:
                up += 1
            else:
                non_up += 1
                char_id = int(generate_rand_float(rng_states, tid) * current_pool_size)
                collection[char_id] += 1
                if collection[char_id] > item_threshold:
                    special_items += 1
            k = 0
            
    final_results[tid, 0] = gold
    final_results[tid, 1] = up
    final_results[tid, 2] = non_up
    final_results[tid, 3] = special_items
    
    for i in range(MAX_UNIQUE_NON_UP_CHARS):
        final_collections[tid, i] = collection[i]

# =============================================================================
# CPU 辅助函数 (数据处理和结果展示)
# =============================================================================

def analyze_and_print_duplicate_stats(collections, simulations):
    """分析并打印非UP角色的重复获取统计。"""
    print("\n⭐ 角色重复获取统计")
    header = f"{'阈值':<12} | {'平均达成数':<12} | " + " | ".join([f"达成{i}个" for i in range(9)])
    print(header); print("-" * len(header))
    for threshold in [5, 6, 7]:
        counts_per_player = np.sum(collections >= threshold, axis=1)
        avg_chars_met = counts_per_player.mean()
        dist = [np.sum(counts_per_player == j) / simulations * 100 for j in range(9)]
        dist_str = " | ".join([f"{p:5.2f}%" for p in dist])
        print(f"获取 {threshold}+ 次 | {avg_chars_met:<12.3f} | {dist_str}")

def save_results_to_file(filename, params, results_data, collections=None):
    """将模拟的详细参数和结果数据保存到文本文件中。"""
    try:
        output_dir = "simdata"; full_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n" + "Gacha Simulation Results\n" + "="*50 + "\n")
            f.write(f"Target Type: {params['target_type']}\n")
            if params['target_type'] == 'pulls':
                f.write(f"Max Pulls per Sim: {int(params['max_pulls']):,}\n")
                f.write(f"Pulls Ratio: {'-'.join(map(str, params['pulls_ratio']))}\n")
            else: f.write(f"Target Count: {int(params['target_count']):,}\n")
            f.write(f"Total Simulations: {int(params['simulations']):,}\n\n")

            for dist_name, data in results_data.items():
                f.write("-" * 50 + f"\nDistribution for: {dist_name}\n" + "-" * 50 + "\n")
                f.write(f"{'Count':<10} | {'Simulations':<15} | {'Percentage (%)'}\n" + "-" * 50 + "\n")
                if len(data) == 0: f.write("No data available.\n\n"); continue
                counts = np.bincount(data)
                for value, num_sims in enumerate(counts):
                    if num_sims > 0: f.write(f"{value:<10} | {int(num_sims):<15,} | {(num_sims/len(data))*100:.8f}%\n")
                f.write("\n")
            
            if collections is not None:
                f.write("="*50 + "\n" + "Duplication Statistics\n" + "="*50 + "\n")
                f.write(f"{'Threshold':<12} | {'Avg Chars Met':<15} | " + " | ".join([f"P({i} Chars Met)" for i in range(9)]) + "\n")
                for threshold in [5, 6, 7]:
                    counts_per_player = np.sum(collections >= threshold, axis=1)
                    avg_chars_met = counts_per_player.mean()
                    dist = [np.sum(counts_per_player == j) / params['simulations'] for j in range(9)]
                    dist_str = " | ".join([f"{p:8.4%}" for p in dist])
                    f.write(f" >= {threshold} copies | {avg_chars_met:<15.4f} | {dist_str}\n")
        print(f"\n💾 详细结果已保存到文件: {full_path}")
    except Exception as e:
        print(f"\n❌ 保存文件时出错: ", e)

def calculate_distribution(data, max_value, interval):
    """使用 numpy 计算数据的直方图分布。"""
    if len(data) == 0: return np.array([]), np.array([])
    max_val_data = data.max() if len(data) > 0 else 0
    max_value = max(max_value, max_val_data)
    bins = np.arange(0, max_value + interval + 1, interval)
    hist, _ = np.histogram(data, bins=bins)
    return hist, bins

def print_distribution(hist, bins, title, unit, show_interval=True):
    """格式化并打印分布数据。"""
    interval_str = f"（{INTERVAL_WIDTH}{unit}）" if show_interval else ""
    print(f"\n📊 {title}分布{interval_str}")
    print("区间范围       | 数量        | 占比 (%)   | 累计 (%)")
    print("------------------------------------------------------")
    total = hist.sum(); cumulative = 0.0
    if total == 0: print("⚠ 无数据"); return
    for i in range(len(hist)):
        if hist[i] == 0: continue
        start, end = bins[i], bins[i+1] - 1
        percentage = hist[i] / total * 100
        cumulative += percentage
        range_str = f"{int(start):4}-{int(end):4}" if start != end else f"{int(start):4}  "
        print(f"{range_str} | {hist[i]:10,} | {percentage:8.6f}% | {cumulative:8.6f}%")

def calculate_statistics(data, total_achieved, max_pulls, percentiles, specific_pulls):
    """计算关键统计数据，如百分位和特定点的概率。"""
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

def run_simulation(target_type, target_count, max_pulls, simulations, batch_size, 
                   pulls_ratio, show_distribution=True, specific_pulls=None, percentiles=None):
    """
    GPU 模拟任务的主控制函数。
    """
    target_count=int(target_count); max_pulls=int(max_pulls); simulations=int(simulations); batch_size=int(batch_size)
    specific_pulls=specific_pulls or []; percentiles=percentiles or []
    
    # --- 'pulls' 模式 (高性能，一次性执行) ---
    if target_type.lower() == 'pulls':
        print(f"▶ 开始GPU“固定抽数”模拟：{simulations:,}个玩家, 每个{max_pulls:,}抽")
        start_time = time.time()
        
        # 定义将在 try 块中使用的变量
        final_results_gpu = None
        final_collections_gpu = None
        
        try:
            # 1. 计算卡池阶段边界
            era_boundaries = np.cumsum(np.round(max_pulls * pulls_ratio / pulls_ratio.sum())).astype(np.int64)
            
            # 2. 分配GPU显存，并提供阶段性进度提示
            final_results_gpu = cuda.device_array((simulations, 4), dtype=np.int64)
            final_collections_gpu = cuda.device_array((simulations, MAX_UNIQUE_NON_UP_CHARS), dtype=np.int32)
            
            # 3. 计算CUDA执行配置
            threads = THREADS_PER_BLOCK
            blocks = math.ceil(simulations / threads)
            
            # 4. 创建随机数状态
            rng_states = create_xoroshiro128p_states(threads * blocks, seed=RAND_SEED)

            # 5. 启动CUDA核函数
            cuda_pulls_mode_kernel[blocks, threads](
                final_results_gpu, final_collections_gpu, max_pulls, era_boundaries, 
                cuda.to_device(ERA_POOL_SIZES), SPECIAL_ITEM_THRESHOLD, 
                simulations, rng_states)
            
            # 6. 等待GPU完成
            cuda.synchronize()

            # 7. 将结果复制回CPU
            final_results = final_results_gpu.copy_to_host()
            final_collections = final_collections_gpu.copy_to_host()

        finally:
            # 8. 【关键】稳健的显存释放
            # 无论 try 块是否成功，都删除对GPU数组的引用，以触发垃圾回收
            del final_results_gpu
            del final_collections_gpu
            
        print(f"▷ 模拟完成 | 耗时 {time.time() - start_time:.1f}s")

        # --- 数据后处理与统计 ---
        golds_data = final_results[:, 0]
        ups_data = final_results[:, 1]
        non_ups_data = final_results[:, 2]
        special_items_data = final_results[:, 3]
        
        total_golds = golds_data.sum()
        total_ups = ups_data.sum()
        total_consumed = simulations * max_pulls
        up_rate = (total_ups / total_golds * 100) if total_golds > 0 else 0
        
        global_stats = {'total_consumed': total_consumed, 'total_gold': total_golds, 'total_up': total_ups, 'up_rate_percent': up_rate, 'combined_avg_gold': (total_consumed / total_golds) if total_golds > 0 else 0, 'combined_avg_up': (total_consumed / total_ups) if total_ups > 0 else 0, 'avg_special_items': special_items_data.mean()}

        # --- 打印结果 (确保所有输出项都存在) ---
        print("\n\n⭐ 全局统计"); print(f"- 综合消耗: {global_stats['total_consumed']:,}抽"); print(f"- 总出金次数: {global_stats['total_gold']:,} (UP率: {global_stats['up_rate_percent']:.6f}%)"); print(f"- 综合平均抽金消耗: {global_stats['combined_avg_gold']:.6f} 抽/金"); print(f"- UP综合平均消耗: {global_stats['combined_avg_up']:.6f} 抽/UP")
        print("\n⭐ 固定抽数模式统计"); print(f"- 平均出金数: {golds_data.mean():.6f}个/次"); print(f"- 平均UP数: {ups_data.mean():.6f}个/次"); print(f"- 平均非UP数: {non_ups_data.mean():.6f}个/次"); print(f"- 平均特殊道具数: {special_items_data.mean():.6f}个/次")
        analyze_and_print_duplicate_stats(final_collections, simulations)

        if show_distribution:
            max_dist_val = max(10, int(max_pulls / 60)) if max_pulls > 0 else 10
            print_distribution(*calculate_distribution(golds_data, max_dist_val, 1), "出金数量", "个", False)
            print_distribution(*calculate_distribution(ups_data, max_dist_val, 1), "UP数量", "个", False)
            print_distribution(*calculate_distribution(non_ups_data, max_dist_val, 1), "非UP数量", "个", False)
            max_special = special_items_data.max() if len(special_items_data) > 0 and special_items_data.max() > 0 else 10
            print_distribution(*calculate_distribution(special_items_data, max(10, int(max_special)+2), 1), "特殊道具数量", "个", False)
            
        filename = f"gachasim_pulls_{max_pulls}_{simulations}.txt"
        params_to_save = {'target_type': target_type, 'max_pulls': max_pulls, 'pulls_ratio': pulls_ratio, 'simulations': simulations}
        results_to_save = {'Gold': golds_data, 'UP': ups_data, 'Non-UP': non_ups_data, 'Special_Items': special_items_data}
        save_results_to_file(filename, params_to_save, results_to_save, collections=final_collections)
        
        return global_stats

    # --- 'gold' / 'up' 模式 (分批次执行，带百分比进度条) ---
    else:
        if target_type.lower() == 'gold': target_mode = 0; actual_max_pulls = target_count * 100
        elif target_type.lower() == 'up': target_mode = 1; actual_max_pulls = target_count * 200
        else: raise ValueError("无效目标类型")
        filename = f"gachasim_{target_type.lower()}_{target_count}_{simulations}.txt"
        print(f"▶ 开始GPU模拟：{simulations:,}次, 目标{target_count}个{target_type}")
        all_results, start_time = [], time.time()
        global_stats = {'total_pulls': 0, 'total_gold_pulls': 0, 'total_up_pulls': 0, 'total_gold': 0, 'total_up': 0, 'total_consumed': 0}
        remaining, completed = simulations, 0
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            # 此处同样可以使用 try...finally 结构来确保每次循环的显存都被释放
            results_gpu = None
            try:
                threads = THREADS_PER_BLOCK; blocks = math.ceil(current_batch / threads)
                results_gpu = cuda.device_array((current_batch, 6), dtype=np.int64)
                rng_states = create_xoroshiro128p_states(threads*blocks, seed=RAND_SEED + completed)
                cuda_gacha_kernel[blocks, threads](results_gpu, actual_max_pulls, target_mode, target_count, current_batch, rng_states)
                cuda.synchronize()
                batch_results = results_gpu.copy_to_host()
            finally:
                del results_gpu

            all_results.append(batch_results)
            global_stats['total_pulls'] += batch_results[:, 0].sum(); global_stats['total_gold'] += batch_results[:, 2].sum()
            global_stats['total_up'] += batch_results[:, 3].sum(); global_stats['total_gold_pulls'] += batch_results[:, 4].sum()
            global_stats['total_up_pulls'] += batch_results[:, 5].sum()
            completed += current_batch; remaining -= current_batch
            print(f"▷ 进度 {100*completed/simulations:.2f}% | 耗时 {time.time() - start_time:.1f}s", end='\r')
        
        
        global_stats['total_consumed'] = global_stats['total_pulls']
        full_results = np.concatenate(all_results); success_mask = full_results[:, 1] == 1; success_count = full_results[:, 1].sum()
        up_rate = (global_stats['total_up'] / global_stats['total_gold'] * 100) if global_stats['total_gold'] > 0 else 0
        actual_avg_gold = global_stats['total_gold_pulls'] / global_stats['total_gold'] if global_stats['total_gold'] > 0 else 0
        combined_avg_gold = global_stats['total_consumed'] / global_stats['total_gold'] if global_stats['total_gold'] > 0 else 0
        actual_avg_up = global_stats['total_up_pulls'] / global_stats['total_up'] if global_stats['total_up'] > 0 else 0
        combined_avg_up = global_stats['total_consumed'] / global_stats['total_up'] if global_stats['total_up'] > 0 else 0
        print("\n\n⭐ 全局统计"); print(f"- 总抽数: {global_stats.get('total_pulls', 0):,}抽"); print(f"- 实际抽金消耗: {global_stats.get('total_gold_pulls', 0):,}抽（最后一次出金总和）"); print(f"- 实际抽UP消耗: {global_stats.get('total_up_pulls', 0):,}抽（最后一次UP总和）"); print(f"- 综合消耗: {global_stats.get('total_consumed', 0):,}抽（与实际消耗相同）"); print(f"- 总出金次数: {global_stats.get('total_gold', 0):,} (UP率: {up_rate:.6f}%)"); print(f"- 实际平均抽金消耗: {actual_avg_gold:.6f} 抽/金"); print(f"- 综合平均抽金消耗: {combined_avg_gold:.6f} 抽/金"); print(f"- UP实际平均消耗: {actual_avg_up:.6f} 抽/UP"); print(f"- UP综合平均消耗: {combined_avg_up:.6f} 抽/UP")
        print(f"\n⭐ 目标达成统计"); print(f"- 成功率: {success_count / simulations * 100:.6f}%")
        if success_count > 0:
            success_pulls = full_results[success_mask, 0]
            print(f"- 平均消耗抽数: {success_pulls.mean():.6f}抽（成功案例）")
            if show_distribution:
                pulls_hist, pulls_bins = calculate_distribution(success_pulls, actual_max_pulls, INTERVAL_WIDTH)
                print_distribution(pulls_hist, pulls_bins, "消耗抽数", "抽")
            stats = calculate_statistics(success_pulls, success_count, actual_max_pulls, percentiles, specific_pulls)
            print("\n📈 关键百分位统计")
            for p, val in stats['percentiles'].items(): print(f"{p}% | {val:8}抽 | 有{p}%的模拟消耗≤此抽数")
        params_to_save = {'target_type': target_type, 'max_pulls': max_pulls, 'target_count': target_count, 'simulations': simulations}
        results_to_save = {}
        if success_count > 0: results_to_save['Pulls_to_achieve_target'] = full_results[success_mask, 0]
        if results_to_save: save_results_to_file(filename, params_to_save, results_to_save)
        return {'total_pulls': global_stats['total_pulls'], 'success_rate': success_count / simulations}


# 当这个脚本作为主程序直接运行时，执行以下代码
if __name__ == "__main__":
    print("--- 脚本独立运行 ---")
    
    run_simulation(
        target_type=TARGET_TYPE,
        target_count=TARGET_COUNT,
        max_pulls=MAX_PULLS,
        simulations=SIMULATIONS,
        batch_size=BATCH_SIZE,
        pulls_ratio=PULLS_RATIO,
        show_distribution=SHOW_DISTRIBUTION,
        specific_pulls=SPECIFIC_PULLS,
        percentiles=PERCENTILES
    )
    
    print("\n--- 模拟完成 ---")