import time
import sys
import numpy as np
import os

# 检查依赖的模拟器文件是否存在
try:
    from gachasim_gpu import run_simulation
except ImportError:
    print("错误：无法导入 'run_simulation' 函数。")
    print("请确保将最新的GPU版本代码保存为 'gachasim_gpu_final.py'，并与此脚本放在同一个文件夹下。")
    sys.exit(1)
except Exception as e:
    print(f"导入 gachasim_gpu_final.py 时发生未知错误: {e}")
    sys.exit(1)

def main():
    """
    自动、循环调用GPU模拟引擎，执行大规模批量模拟任务。
    """
    # --- 批处理任务配置 ---
    start_pulls = 100
    end_pulls = 10000
    step_pulls = 10
    
    # 模拟参数 (请注意，这是一个极大的计算量)
    simulations_per_run = 10_000_000
    batch_size_per_run = 100_000
    
    # 固定参数
    pulls_ratio_for_runs = np.array([4, 1, 4, 1])
    target_type_for_runs = 'pulls'

    total_start_time = time.time()
    print("--- 开始大规模批量模拟任务 ---")
    print(f"模拟范围: 从 {start_pulls} 抽到 {end_pulls} 抽，步长为 {step_pulls} 抽。")
    print(f"每次运行的模拟次数: {simulations_per_run:,}")
    print(f"每次运行的批处理大小: {batch_size_per_run:,}")
    print("结果将自动保存到 'simdata' 文件夹中。")
    print("-" * 50)

    # 生成任务列表
    run_list = list(range(start_pulls, end_pulls + 1, step_pulls))
    total_runs = len(run_list)

    for i, current_max_pulls in enumerate(run_list):
        print(f"\n--- 任务 ({i+1}/{total_runs}): 开始模拟 MAX_PULLS = {current_max_pulls} ---")
        run_start_time = time.time()

        try:
            # 调用从 gachasim_gpu_final.py 导入的函数
            # 设置 show_distribution=0 以避免在终端打印海量分布图
            run_simulation(
                target_type=target_type_for_runs,
                target_count=1,  # 'pulls'模式下未使用，仅作占位
                max_pulls=current_max_pulls,
                simulations=simulations_per_run,
                batch_size=batch_size_per_run,
                pulls_ratio=pulls_ratio_for_runs,
                show_distribution=0 
            )
            run_end_time = time.time()
            print(f"--- 任务 ({i+1}/{total_runs}): 完成，耗时 {run_end_time - run_start_time:.2f} 秒 ---")

        except Exception as e:
            print(f"--- 任务 ({i+1}/{total_runs}): 在 MAX_PULLS = {current_max_pulls} 时发生严重错误，任务已终止 ---")
            print(f"错误详情: {e}")
            # 引入 traceback 模块以打印更详细的错误堆栈
            import traceback
            traceback.print_exc()
            break # 如果某个任务出错，则终止整个批处理

    total_end_time = time.time()
    successful_runs = i + 1 if 'i' in locals() else 0
    print("\n" + "=" * 50)
    print("--- 所有批量模拟任务已执行完毕 ---")
    print(f"成功运行任务数: {successful_runs}/{total_runs}")
    print(f"总耗时: {(total_end_time - total_start_time) / 3600:.2f} 小时")
    print("=" * 50)

if __name__ == "__main__":
    main()