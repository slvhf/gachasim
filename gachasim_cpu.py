# 导入必要的库
import numpy as np
from numba import jit, prange  # Numba 用于即时编译加速代码
import time
import math
import warnings
import os

# --- 错误修复 ---
# 尝试忽略特定类型的警告，以保持输出整洁。
# 新版本的 numpy (>=2.0) 中 np.VisibleDeprecationWarning 已被移除，
# 使用 try-except 语句块可以兼容新旧版本的 numpy。
try:
    warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning)
except AttributeError:
    pass  # 如果属性不存在（新版numpy），则静默跳过

# ####################################
# 用户可修改参数 (当此脚本独立运行时使用)
# ####################################

# --- 模拟模式配置 ---
# 'pulls': 固定抽数模式，模拟在 MAX_PULLS 抽内能获得什么。
# 'gold': 目标为 N 个五星角色。
# 'up':   目标为 N 个 UP 五星角色。
TARGET_TYPE = 'pulls'
TARGET_COUNT = 1      # 在 'gold' 或 'up' 模式下，要达成的目标数量
MAX_PULLS = 5000      # 在 'pulls' 模式下，每个玩家的总抽数

# --- 性能与精度配置 ---
SIMULATIONS = 100_000   # 总模拟次数（可以理解为模拟的玩家数量）
BATCH_SIZE = 5_000      # 批处理大小，每次处理多少个模拟实例（主要用于 'gold'/'up' 模式）

# --- 'pulls' 模式专属配置 ---
# 定义不同阶段的抽数比例，用于模拟动态卡池
# 例如 [4, 1, 4, 1] 表示总抽数按 4:1:4:1 的比例分配到四个阶段
PULLS_RATIO = np.array([4, 1, 4, 1])

# --- 统计输出配置 ---
# 在 'gold'/'up' 模式下，计算在这些特定抽数点达成目标的概率
SPECIFIC_PULLS = [260, 320, 940]
# 在 'gold'/'up' 模式下，计算达成这些百分位所需的抽数
PERCENTILES = [10, 25, 50, 75, 90]
INTERVAL_WIDTH = 1    # 打印分布图时，每个区间的宽度
SHOW_DISTRIBUTION = 1 # 是否在控制台打印详细的分布图 (1 for Yes, 0 for No)

# ####################################
# 核心模拟常量 (通常不需要修改)
# ####################################

# 'pulls' 模式下的卡池配置
# ERA_POOL_SIZES 定义了四个不同阶段的非UP角色卡池大小
ERA_POOL_SIZES = np.array([5, 6, 7, 8])
# 特殊道具阈值：当某个非UP角色获取次数超过此阈值时，计为一个特殊道具
SPECIAL_ITEM_THRESHOLD = 7
# 收藏柜中最多能容纳的独立非UP角色数量
MAX_UNIQUE_NON_UP_CHARS = 15


@jit(nopython=True, parallel=True, fastmath=True)
def cpu_gacha_batch(target_mode, target_count, max_pulls, batch_size):
    """
    为 'gold' 和 'up' 模式设计的CPU JIT加速函数。
    此函数通过并行处理，一次性执行 `batch_size` 个模拟实例。

    参数:
    - target_mode (int): 目标模式, 0 代表 'gold', 1 代表 'up'。
    - target_count (int): 目标五星或UP的数量。
    - max_pulls (int): 单次模拟的最大抽数上限，防止无限循环。
    - batch_size (int): 此批次需要执行的模拟总数。

    返回:
    - results (np.ndarray): 一个 (batch_size, 6) 的数组，记录了每次模拟的结果。
        - results[:, 0]: 总消耗抽数
        - results[:, 1]: 是否达成目标 (1 for Yes, 0 for No)
        - results[:, 2]: 获得的总五星数
        - results[:, 3]: 获得的总UP数
        - results[:, 4]: 最后一次出金时的总抽数
        - results[:, 5]: 最后一次出UP时的总抽数
    """
    # 初始化结果数组，用于存储每条线程的模拟结果
    results = np.zeros((batch_size, 6), dtype=np.int64)
    
    # prange 是 numba 提供的并行循环
    for i in prange(batch_size):
        # --- 单次模拟状态变量初始化 ---
        k = 0                  # 当前的垫刀次数（自上次出金后的抽数）
        gold = 0               # 已获得五星角色总数
        up = 0                 # 已获得UP角色总数
        pulls_used = 0         # 已消耗的总抽数
        ming_guang_counter = 1 # 特殊机制计数器
        guaranteed_up = False  # 是否处于大保底状态
        achieved = False       # 是否已达成目标
        last_gold = 0          # 最后一次出金时的总抽数
        last_up = 0            # 最后一次出UP时的总抽数

        # 循环抽卡，直到达成目标或达到抽数上限
        while not (achieved or pulls_used >= max_pulls):
            k += 1  # 垫刀次数+1
            
            # --- 计算当前抽的出金概率 ---
            threshold = 600  # 基础概率 0.6% (600 / 100,000)
            if k >= 74:      # 软保底机制：从第74抽开始，概率线性增加
                threshold = min(600 + 6000 * (k - 73), 100000)

            # --- 判定是否出金 ---
            if np.random.randint(0, 100000) < threshold:
                pulls_used += k   # 更新总消耗抽数
                gold += 1         # 五星总数+1
                last_gold = pulls_used # 记录出金时刻
                is_up = False     # 本次出金是否为UP，先默认为False

                # --- 判定是否为UP角色（大小保底机制） ---
                if guaranteed_up:
                    is_up = True
                    guaranteed_up = False
                else:
                    if ming_guang_counter >= 3: # 特殊机制：计数器满3必定为UP
                        is_up = True
                        ming_guang_counter = 1
                    else:
                        if np.random.randint(0, 1000) < 500: # 50/50 判定
                            is_up = True
                            ming_guang_counter = max(0, ming_guang_counter - 1)
                        else: # 歪了
                            guaranteed_up = True
                            ming_guang_counter = min(3, ming_guang_counter + 1)
                
                if is_up:
                    up += 1        # UP总数+1
                    last_up = pulls_used # 记录出UP时刻
                
                k = 0 # 出金后，垫刀次数清零

            # --- 检查是否达成目标 ---
            is_gold_mode_achieved = (target_mode == 0 and gold >= target_count)
            is_up_mode_achieved = (target_mode == 1 and up >= target_count)
            if is_gold_mode_achieved or is_up_mode_achieved:
                achieved = True

        # --- 保存本次模拟的结果 ---
        results[i, 0] = np.int64(pulls_used)
        results[i, 1] = np.int64(1 if achieved else 0)
        results[i, 2] = np.int64(gold)
        results[i, 3] = np.int64(up)
        results[i, 4] = np.int64(last_gold)
        results[i, 5] = np.int64(last_up)
        
    return results

@jit(nopython=True, parallel=True, fastmath=True)
def cpu_pulls_mode_simulation(simulations, max_pulls, era_boundaries, pool_sizes, item_threshold):
    """
    为 'pulls' 模式设计的CPU JIT加速函数。
    模拟 `simulations` 个玩家，每人抽 `max_pulls` 次，并统计最终收益和收藏。

    参数:
    - simulations (int): 模拟的玩家总数。
    - max_pulls (int): 每个玩家的总抽数。
    - era_boundaries (np.ndarray): 划分卡池阶段的抽数边界。
    - pool_sizes (np.ndarray): 每个阶段的非UP卡池大小。
    - item_threshold (int): 判定为“特殊道具”的重复角色阈值。

    返回:
    - final_results (np.ndarray): (simulations, 4) 数组，记录每个玩家的最终统计。
        - [:, 0]: 获得的总五星数
        - [:, 1]: 获得的总UP数
        - [:, 2]: 获得的总非UP数
        - [:, 3]: 获得的特殊道具数
    - final_collections (np.ndarray): (simulations, MAX_UNIQUE_NON_UP_CHARS) 数组，
                                       记录每个玩家的非UP角色收藏情况。
    """
    # 初始化结果数组
    final_results = np.zeros((simulations, 4), dtype=np.int64)
    final_collections = np.zeros((simulations, MAX_UNIQUE_NON_UP_CHARS), dtype=np.int32)

    # 并行模拟每个玩家
    for i in prange(simulations):
        # --- 单个玩家状态变量初始化 ---
        k, gold, up, non_up, special_items = 0, 0, 0, 0, 0
        guaranteed_up = False
        ming_guang_counter = 1
        collection = np.zeros(MAX_UNIQUE_NON_UP_CHARS, dtype=np.int32) # 个人收藏柜
        
        # 循环抽卡，直到抽满 max_pulls 次
        for pull in range(1, max_pulls + 1):
            # --- 确定当前抽数所处的卡池阶段 ---
            era_idx = 0
            for boundary_idx in range(len(era_boundaries) - 1):
                if pull > era_boundaries[boundary_idx]:
                    era_idx = boundary_idx + 1
            current_pool_size = pool_sizes[era_idx]

            # --- 概率和出金判定 (逻辑同上一个函数) ---
            k += 1
            threshold = 600
            if k >= 74:
                threshold = min(600 + 6000 * (k - 73), 100000)

            if np.random.randint(0, 100000) < threshold:
                is_up = False
                gold += 1
                
                # 大小保底判定
                if guaranteed_up:
                    is_up, guaranteed_up = True, False
                else:
                    if ming_guang_counter >= 3:
                        is_up, ming_guang_counter = True, 1
                    else:
                        if np.random.randint(0, 1000) < 500:
                            is_up, ming_guang_counter = True, max(0, ming_guang_counter - 1)
                        else:
                            guaranteed_up, ming_guang_counter = True, min(3, ming_guang_counter + 1)
                
                # 根据是否为UP，更新不同计数器
                if is_up:
                    up += 1
                else:
                    non_up += 1
                    # 从当前阶段的卡池中随机抽取一个非UP角色
                    char_id = np.random.randint(0, current_pool_size)
                    collection[char_id] += 1
                    # 检查是否因为重复获取而得到特殊道具
                    if collection[char_id] > item_threshold:
                        special_items += 1
                k = 0 # 重置垫刀次数
                
        # --- 保存该玩家的最终模拟结果 ---
        # 拆分赋值以更好地兼容Numba的并行编译器
        final_results[i, 0] = gold
        final_results[i, 1] = up
        final_results[i, 2] = non_up
        final_results[i, 3] = special_items

        for j in range(MAX_UNIQUE_NON_UP_CHARS):
            final_collections[i, j] = collection[j]
            
    return final_results, final_collections

def analyze_and_print_duplicate_stats(collections, simulations):
    """
    分析并打印非UP角色的重复获取统计（例如有多少个角色抽到了5次以上）。
    """
    print("\n⭐ 角色重复获取统计")
    header = f"{'阈值':<12} | {'平均达成数':<12} | " + " | ".join([f"达成{i}个" for i in range(9)])
    print(header)
    print("-" * len(header))
    
    for threshold in [5, 6, 7]:
        # 对每个玩家，计算有多少个角色的获取次数超过了阈值
        counts_per_player = np.sum(collections >= threshold, axis=1)
        # 计算所有玩家的平均值
        avg_chars_met = counts_per_player.mean()
        # 计算达成0个、1个、2个...的玩家占比
        dist = [np.sum(counts_per_player == j) / simulations * 100 for j in range(9)]
        dist_str = " | ".join([f"{p:5.2f}%" for p in dist])
        print(f"获取 {threshold}+ 次 | {avg_chars_met:<12.3f} | {dist_str}")

def save_results_to_file(filename, params, results_data, collections=None):
    """
    将模拟的详细参数和结果数据保存到文本文件中。
    """
    try:
        output_dir = "simdata"
        full_path = os.path.join(output_dir, filename)
        
        # 如果文件夹不存在，则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(full_path, 'w', encoding='utf-8') as f:
            # 写入文件头和模拟参数
            f.write("="*50 + "\n" + "Gacha Simulation Results\n" + "="*50 + "\n")
            f.write(f"Target Type: {params['target_type']}\n")
            if params['target_type'] == 'pulls':
                f.write(f"Max Pulls per Sim: {int(params['max_pulls']):,}\n")
                f.write(f"Pulls Ratio: {'-'.join(map(str, params['pulls_ratio']))}\n")
            else:
                f.write(f"Target Count: {int(params['target_count']):,}\n")
            f.write(f"Total Simulations: {int(params['simulations']):,}\n\n")

            # 写入各种结果的详细分布
            for dist_name, data in results_data.items():
                f.write("-" * 50 + f"\nDistribution for: {dist_name}\n" + "-" * 50 + "\n")
                f.write(f"{'Count':<10} | {'Simulations':<15} | {'Percentage (%)'}\n" + "-" * 50 + "\n")
                if len(data) == 0:
                    f.write("No data available.\n\n")
                    continue
                counts = np.bincount(data)
                for value, num_sims in enumerate(counts):
                    if num_sims > 0:
                        f.write(f"{value:<10} | {int(num_sims):<15,} | {(num_sims/len(data))*100:.8f}%\n")
                f.write("\n")
            
            # 如果有收藏数据，则写入重复统计
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
    """
    使用 numpy 计算数据的直方图分布。
    """
    if len(data) == 0:
        return np.array([]), np.array([])
    
    max_val_data = data.max() if len(data) > 0 else 0
    # 确保箱子的上界足够大
    max_value = max(max_value, max_val_data)
    # 创建直方图的箱子边界
    bins = np.arange(0, max_value + interval + 1, interval)
    hist, _ = np.histogram(data, bins=bins)
    
    return hist, bins

def print_distribution(hist, bins, title, unit, show_interval=True):
    """
    格式化并打印分布数据。
    """
    interval_str = f"（{INTERVAL_WIDTH}{unit}）" if show_interval else ""
    print(f"\n📊 {title}分布{interval_str}")
    print("区间范围       | 数量        | 占比 (%)   | 累计 (%)")
    print("------------------------------------------------------")
    
    total = hist.sum()
    if total == 0:
        print("⚠ 无数据")
        return
        
    cumulative = 0.0
    for i in range(len(hist)):
        if hist[i] == 0:
            continue
            
        start = bins[i]
        end = bins[i+1] - 1
        percentage = (hist[i] / total) * 100
        cumulative += percentage
        
        range_str = f"{int(start):4}-{int(end):4}" if start != end else f"{int(start):4}  "
        print(f"{range_str} | {hist[i]:10,} | {percentage:8.6f}% | {cumulative:8.6f}%")

def calculate_statistics(data, total_achieved, max_pulls, percentiles, specific_pulls):
    """
    计算关键统计数据，如百分位和特定点的概率。
    """
    stats = {'percentiles': {}, 'specific_pulls': {}, 'min': 0, 'max': 0}
    if total_achieved == 0 or len(data) == 0:
        return stats
        
    sorted_data = np.sort(data)
    stats['min'] = sorted_data[0]
    stats['max'] = sorted_data[-1]
    
    # 计算百分位
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    for p in percentiles:
        # 找到第一个累计概率 >= p 的位置
        index = np.searchsorted(cumulative, p)
        # 边界检查
        index = min(max(index, 0), len(sorted_data) - 1)
        stats['percentiles'][p] = sorted_data[index]
        
    # 计算特定抽数点的达成概率
    for pull in specific_pulls:
        count_le_pull = np.sum(sorted_data <= pull)
        stats['specific_pulls'][pull] = (count_le_pull / len(sorted_data)) * 100
        
    return stats

def run_simulation(target_type, target_count, max_pulls, simulations, batch_size, 
                   pulls_ratio, show_distribution=True, specific_pulls=None, percentiles=None):
    """
    模拟任务的主控制函数。
    它根据 `target_type` 来决定调用哪个模拟核心，并负责处理数据和打印结果。
    """
    
    # --- 参数初始化和类型转换 ---
    target_count = int(target_count)
    max_pulls = int(max_pulls)
    simulations = int(simulations)
    batch_size = int(batch_size)
    specific_pulls = specific_pulls or []
    percentiles = percentiles or []
    
    # --- 分支：'pulls' 模式 ---
    if target_type.lower() == 'pulls':
        print(f"▶ 开始CPU“固定抽数”模拟：{simulations:,}个玩家, 每个{max_pulls:,}抽")
        start_time = time.time()
        
        # 根据抽数比例计算卡池阶段的边界
        era_boundaries = np.cumsum(
            np.round(max_pulls * pulls_ratio / pulls_ratio.sum())
        ).astype(np.int64)
        
        # 调用 JIT 加速的核心模拟函数
        final_results, final_collections = cpu_pulls_mode_simulation(
            simulations, max_pulls, era_boundaries, ERA_POOL_SIZES, SPECIAL_ITEM_THRESHOLD)
        
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
        
        global_stats = {
            'total_consumed': total_consumed,
            'total_gold': total_golds,
            'total_up': total_ups,
            'up_rate_percent': up_rate,
            'combined_avg_gold': (total_consumed / total_golds) if total_golds > 0 else 0,
            'combined_avg_up': (total_consumed / total_ups) if total_ups > 0 else 0,
            'avg_special_items': special_items_data.mean()
        }

        # --- 打印结果 ---
        print("\n\n⭐ 全局统计")
        print(f"- 综合消耗: {global_stats['total_consumed']:,}抽")
        print(f"- 总出金次数: {global_stats['total_gold']:,} (UP率: {global_stats['up_rate_percent']:.6f}%)")
        print(f"- 综合平均抽金消耗: {global_stats['combined_avg_gold']:.6f} 抽/金")
        print(f"- UP综合平均消耗: {global_stats['combined_avg_up']:.6f} 抽/UP")
        
        print("\n⭐ 固定抽数模式统计")
        print(f"- 平均出金数: {golds_data.mean():.6f}个/次")
        print(f"- 平均UP数: {ups_data.mean():.6f}个/次")
        print(f"- 平均非UP数: {non_ups_data.mean():.6f}个/次")
        print(f"- 平均特殊道具数: {special_items_data.mean():.6f}个/次")
        
        analyze_and_print_duplicate_stats(final_collections, simulations)

        if show_distribution:
            max_dist_val = max(10, int(max_pulls / 60)) if max_pulls > 0 else 10
            print_distribution(*calculate_distribution(golds_data, max_dist_val, 1), "出金数量", "个", False)
            print_distribution(*calculate_distribution(ups_data, max_dist_val, 1), "UP数量", "个", False)
            print_distribution(*calculate_distribution(non_ups_data, max_dist_val, 1), "非UP数量", "个", False)
            max_special_items = special_items_data.max() if len(special_items_data) > 0 and special_items_data.max() > 0 else 10
            print_distribution(*calculate_distribution(special_items_data, max(10, int(max_special_items)+2), 1), "特殊道具数量", "个", False)
            
        # --- 保存结果到文件 ---
        filename = f"gachasim_pulls_{max_pulls}_{simulations}.txt"
        params_to_save = {
            'target_type': target_type,
            'max_pulls': max_pulls,
            'pulls_ratio': pulls_ratio,
            'simulations': simulations
        }
        results_to_save = {
            'Gold': golds_data,
            'UP': ups_data,
            'Non-UP': non_ups_data,
            'Special_Items': special_items_data
        }
        save_results_to_file(filename, params_to_save, results_to_save, collections=final_collections)
        
        return global_stats

    # --- 分支：'gold' 或 'up' 模式 ---
    else:
        if target_type.lower() == 'gold':
            target_mode = 0
            actual_max_pulls = target_count * 100 # 设置一个合理的抽数上限
        elif target_type.lower() == 'up':
            target_mode = 1
            actual_max_pulls = target_count * 200 # UP期望更高，所以上限也更高
        else:
            raise ValueError("无效目标类型, 请选择 'pulls', 'gold', 或 'up'")
        
        filename = f"gachasim_{target_type.lower()}_{target_count}_{simulations}.txt"
        print(f"▶ 开始CPU模拟：{simulations:,}次, 目标{target_count}个{target_type}")
        
        all_results = []
        start_time = time.time()
        
        # 初始化全局统计数据字典
        global_stats = {
            'total_pulls': 0, 'total_gold_pulls': 0, 'total_up_pulls': 0,
            'total_gold': 0, 'total_up': 0, 'total_consumed': 0
        }
        
        remaining = simulations
        # --- 分批次执行模拟 ---
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            
            # 调用JIT核心函数
            batch_results = cpu_gacha_batch(target_mode, target_count, actual_max_pulls, current_batch)
            all_results.append(batch_results)
            
            # --- 实时累加统计数据 ---
            global_stats['total_pulls'] += batch_results[:, 0].sum()
            global_stats['total_gold'] += batch_results[:, 2].sum()
            global_stats['total_up'] += batch_results[:, 3].sum()
            global_stats['total_gold_pulls'] += batch_results[:, 4].sum()
            global_stats['total_up_pulls'] += batch_results[:, 5].sum()
            
            remaining -= current_batch
            
            # 打印进度条
            progress = 100 * (simulations - remaining) / simulations
            elapsed_time = time.time() - start_time
            print(f"▷ 进度 {progress:.2f}% | 耗时 {elapsed_time:.1f}s", end='\r')
            
        global_stats['total_consumed'] = global_stats['total_pulls']
        
        # --- 整合所有批次的结果 ---
        full_results = np.concatenate(all_results)
        success_mask = (full_results[:, 1] == 1)
        success_count = full_results[:, 1].sum()
        
        # --- 计算最终统计指标 ---
        up_rate = (global_stats['total_up'] / global_stats['total_gold'] * 100) if global_stats['total_gold'] > 0 else 0
        actual_avg_gold = global_stats['total_gold_pulls'] / global_stats['total_gold'] if global_stats['total_gold'] > 0 else 0
        combined_avg_gold = global_stats['total_consumed'] / global_stats['total_gold'] if global_stats['total_gold'] > 0 else 0
        actual_avg_up = global_stats['total_up_pulls'] / global_stats['total_up'] if global_stats['total_up'] > 0 else 0
        combined_avg_up = global_stats['total_consumed'] / global_stats['total_up'] if global_stats['total_up'] > 0 else 0

        # --- 打印结果 ---
        print("\n\n⭐ 全局统计")
        print(f"- 总抽数: {global_stats.get('total_pulls', 0):,}抽")
        print(f"- 实际抽金消耗: {global_stats.get('total_gold_pulls', 0):,}抽（最后一次出金总和）")
        print(f"- 实际抽UP消耗: {global_stats.get('total_up_pulls', 0):,}抽（最后一次UP总和）")
        print(f"- 综合消耗: {global_stats.get('total_consumed', 0):,}抽（与实际消耗相同）")
        print(f"- 总出金次数: {global_stats.get('total_gold', 0):,} (UP率: {up_rate:.6f}%)")
        print(f"- 实际平均抽金消耗: {actual_avg_gold:.6f} 抽/金")
        print(f"- 综合平均抽金消耗: {combined_avg_gold:.6f} 抽/金")
        print(f"- UP实际平均消耗: {actual_avg_up:.6f} 抽/UP")
        print(f"- UP综合平均消耗: {combined_avg_up:.6f} 抽/UP")
        
        print(f"\n⭐ 目标达成统计")
        print(f"- 成功率: {success_count / simulations * 100:.6f}%")
        
        if success_count > 0:
            success_pulls = full_results[success_mask, 0]
            print(f"- 平均消耗抽数: {success_pulls.mean():.6f}抽（仅统计成功案例）")
            
            if show_distribution:
                pulls_hist, pulls_bins = calculate_distribution(success_pulls, actual_max_pulls, INTERVAL_WIDTH)
                print_distribution(pulls_hist, pulls_bins, "消耗抽数", "抽")
                
            stats = calculate_statistics(success_pulls, success_count, actual_max_pulls, percentiles, specific_pulls)
            print("\n📈 关键百分位统计")
            for p, val in stats['percentiles'].items():
                print(f"{p}% | {val:8}抽 | 有{p}%的模拟消耗≤此抽数")
        
        # --- 保存结果到文件 ---
        params_to_save = {
            'target_type': target_type,
            'max_pulls': max_pulls,
            'target_count': target_count,
            'simulations': simulations
        }
        results_to_save = {}
        if success_count > 0:
            results_to_save['Pulls_to_achieve_target'] = full_results[success_mask, 0]
            
        if results_to_save:
            save_results_to_file(filename, params_to_save, results_to_save)
            
        return {'total_pulls': global_stats['total_pulls'], 'success_rate': success_count / simulations}


# 当这个脚本作为主程序直接运行时，执行以下代码
if __name__ == "__main__":
    print("--- 脚本独立运行 ---")
    
    # 调用主函数，并传入文件顶部的用户配置参数
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