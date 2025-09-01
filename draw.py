import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

def generate_all_plots(csv_filepath):
    """
    读取 simulation_summary.csv 文件并生成所有图表。
    此为包含所有修正的最终版本。
    """
    if not os.path.exists(csv_filepath):
        print(f"错误: 找不到CSV文件 '{csv_filepath}'。请先运行 analysis.py。")
        return

    df = pd.read_csv(csv_filepath)
    print("成功读取 'simulation_summary.csv'，开始生成图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- 1. 期望值图 (已加回) ---
    plt.figure(figsize=(12, 8))
    plt.plot(df['max_pulls'], df['expected_special_items'], marker='o', linestyle='-', markersize=4)
    plt.title('Expected Number of Special Items vs. Max Pulls')
    plt.xlabel('Max Pulls')
    plt.ylabel('Expected Special Items')
    plt.tight_layout()
    plt.savefig('expected_special_items_plot.png')
    plt.close()

    # --- 2. 特殊道具/溢出 AT MOST (<=) ---
    plt.figure(figsize=(12, 8))
    for k in [0, 1, 2, 3, 6, 9, 12]:
        col = f'si_p(<={k})'
        if col in df.columns:
            plt.plot(df['max_pulls'], df[col], marker='.', linestyle='-', label=f'<= {k}')
    plt.title('Probability of Getting AT MOST N Special Items')
    plt.xlabel('Max Pulls')
    plt.ylabel('Probability (%)')
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(100.0))
    plt.ylim(-5, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig('special_items_at_most_prob.png')
    plt.close()

    # --- 3. 特殊道具/溢出 AT LEAST (>=) ---
    plt.figure(figsize=(12, 8))
    for k in [1, 2, 3, 6, 9, 12]:
        col = f'si_p(>={k})'
        if col in df.columns:
            plt.plot(df['max_pulls'], df[col], marker='.', linestyle='-', label=f'>= {k}')
    plt.title('Probability of Getting AT LEAST N Special Items')
    plt.xlabel('Max Pulls')
    plt.ylabel('Probability (%)')
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(100.0))
    plt.ylim(-5, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig('special_items_at_least_prob.png')
    plt.close()

    # --- 4. 重复角色 AT LEAST (>=) ---
    for threshold in [5, 6, 7]:
        plt.figure(figsize=(12, 8))
        
        # 检查是否存在该阈值的数据
        if not any(f'p(0)_{threshold}c' in s for s in df.columns):
            print(f"警告: 阈值为 {threshold} 的副本数据缺失，跳过该图表。")
            plt.close()
            continue

        # 绘制 P(0)
        if f'p(0)_{threshold}c' in df.columns:
            plt.plot(df['max_pulls'], df[f'p(0)_{threshold}c'], marker='.', linestyle='-', label='0 ')
        
        # 绘制 P(>=1) 到 P(>=8)
        for i in range(1, 9):
            col = f'p(>={i})_{threshold}c'
            if col in df.columns and not df[col].isnull().all():
                plt.plot(df['max_pulls'], df[col], marker='.', linestyle='-', label=f'>= {i} ')

        plt.xlabel('Max Pulls')
        plt.ylabel('Probability (%)')
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(100.0))
        plt.ylim(-5, 105)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'duplication_at_least_prob_thresh_{threshold}.png')
        plt.close()

    # --- 5. 重复角色 AT MOST (<=) ---
    for threshold in [5, 6, 7]:
        plt.figure(figsize=(12, 8))
        
        if not any(f'p(0)_{threshold}c' in s for s in df.columns):
            plt.close()
            continue
        
        # 绘制 P(<=0)，即 P(0)
        if f'p(0)_{threshold}c' in df.columns:
             plt.plot(df['max_pulls'], df[f'p(0)_{threshold}c'], marker='.', linestyle='-', label='<= 0 ')
        
        # 绘制 P(<=1) 到 P(<=8)
        for i in range(1, 9):
            col = f'p(<={i})_{threshold}c'
            if col in df.columns and not df[col].isnull().all():
                plt.plot(df['max_pulls'], df[col], marker='.', linestyle='-', label=f'<= {i} ')

        plt.xlabel('Max Pulls')
        plt.ylabel('Probability (%)')
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(100.0))
        plt.ylim(-5, 105)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'duplication_at_most_prob_thresh_{threshold}.png')
        plt.close()
        
    print("所有图表已成功生成。")

if __name__ == '__main__':
    generate_all_plots('simulation_summary.csv')