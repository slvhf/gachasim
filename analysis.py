import os
import re
import pandas as pd

def parse_simulation_file(filepath):
    """
    解析单个模拟结果文件。
    修正了特殊道具概率计算的逻辑错误。
    """
    match = re.search(r'pulls_(\d+)_', os.path.basename(filepath))
    if not match: return None
    max_pulls = int(match.group(1))

    data = {'max_pulls': max_pulls}
    in_special_items_section = False
    in_duplication_section = False
    
    special_items_dist = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Distribution for: Special_Items' in line:
                in_special_items_section = True
                in_duplication_section = False
                continue 
            elif 'Duplication Statistics' in line:
                in_duplication_section = True
                in_special_items_section = False
                continue
            elif 'Distribution for:' in line and 'Special_Items' not in line:
                in_special_items_section = False

            if in_special_items_section:
                if '|' in line and not "Count" in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2 and parts[0].isdigit():
                        try:
                            special_items_dist.append((int(parts[0]), int(parts[1].replace(',', ''))))
                        except (ValueError, IndexError):
                            continue
            
            elif in_duplication_section:
                dup_match = re.search(r'>=\s*(\d)\s*copies', line)
                if dup_match:
                    threshold = int(dup_match.group(1))
                    if threshold in [5, 6, 7]:
                        numbers_found = re.findall(r'(\d+\.\d+)', line)
                        if len(numbers_found) >= 2:
                            try:
                                all_probs = [float(p) for p in numbers_found[1:]]
                                p_ge = [0.0] * len(all_probs)
                                if all_probs:
                                    p_ge[-1] = all_probs[-1]
                                    for i in range(len(all_probs) - 2, -1, -1): p_ge[i] = all_probs[i] + p_ge[i+1]
                                
                                p_le = [0.0] * len(all_probs)
                                if all_probs:
                                    p_le[0] = all_probs[0]
                                    for i in range(1, len(all_probs)): p_le[i] = p_le[i-1] + all_probs[i]

                                data[f'p(0)_{threshold}c'] = all_probs[0] if len(all_probs) > 0 else 0.0
                                for i in range(1, len(all_probs)):
                                    data[f'p(>={i})_{threshold}c'] = p_ge[i]
                                    data[f'p(<={i})_{threshold}c'] = p_le[i]
                            except (ValueError, IndexError):
                                continue
    
    # --- Part 1 Calculation (Special Items / 溢出) ---
    if special_items_dist:
        total_sims = sum(s for c, s in special_items_dist)
        if total_sims > 0:
            data['expected_special_items'] = sum(c * s for c, s in special_items_dist) / total_sims
            dist_df = pd.DataFrame(special_items_dist, columns=['count', 'num_sims']).set_index('count')
            
            # 修正：创建完整的索引以正确处理缺失的计数值
            max_observed_count = dist_df.index.max()
            full_index = pd.Index(range(max_observed_count + 1))
            dist_df = dist_df.reindex(full_index, fill_value=0)
            
            cumulative_sims = dist_df['num_sims'].cumsum()
            
            for k in [0, 1, 2, 3, 6, 9, 12]:
                # --- P(<=k) Calculation Logic ---
                if k > max_observed_count:
                    sims_le_k = total_sims # 如果k超出观测范围，P(<=k)为100%
                else:
                    sims_le_k = cumulative_sims[k]
                data[f'si_p(<={k})'] = (sims_le_k / total_sims) * 100
                
                # --- P(>=k) Calculation Logic ---
                if k > 0:
                    k_minus_1 = k - 1
                    if k_minus_1 > max_observed_count:
                        sims_le_k_minus_1 = total_sims # 如果k-1超出范围, P(<=k-1)为100%
                    else:
                        sims_le_k_minus_1 = cumulative_sims[k_minus_1]
                    # P(>=k) = 100% - P(<k) = 100% - P(<=k-1)
                    data[f'si_p(>={k})'] = 100.0 - ((sims_le_k_minus_1 / total_sims) * 100)

    else: # Fallback if no special items data
        data['expected_special_items'] = 0
        for k in [0, 1, 2, 3, 6, 9, 12]: data[f'si_p(<={k})'] = 100.0
        for k in [1, 2, 3, 6, 9, 12]: data[f'si_p(>={k})'] = 0.0
        
    return data

def main():
    simdata_dir = 'simdata'
    if not os.path.isdir(simdata_dir):
        print(f"错误: 找不到 '{simdata_dir}' 文件夹。")
        return
        
    txt_files = [f for f in os.listdir(simdata_dir) if f.startswith('gachasim_pulls_') and f.endswith('.txt')]
    if not txt_files:
        print(f"错误: 在 '{simdata_dir}' 文件夹中找不到 'gachasim_pulls_*.txt' 文件。")
        return

    all_data = []
    sorted_files = sorted(txt_files, key=lambda f: int(re.search(r'pulls_(\d+)_', f).group(1)))
    
    print(f"开始解析 {len(sorted_files)} 个文件...")
    for filename in sorted_files:
        all_data.append(parse_simulation_file(os.path.join(simdata_dir, filename)))
        
    print(f"成功解析 {len(all_data)} 个文件。")
    df = pd.DataFrame([d for d in all_data if d is not None])
    df.rename(columns={'si_p(<=0)': 'si_p(0)'}, inplace=True, errors='ignore')
    
    expected_cols = ['max_pulls', 'expected_special_items']
    for k in [0, 1, 2, 3, 6, 9, 12]: expected_cols.append(f'si_p(<={k})')
    for k in [1, 2, 3, 6, 9, 12]: expected_cols.append(f'si_p(>={k})')
    for t in [5, 6, 7]:
        expected_cols.append(f'p(0)_{t}c')
        for i in range(1, 9): expected_cols.append(f'p(>={i})_{t}c')
        for i in range(1, 9): expected_cols.append(f'p(<={i})_{t}c')
            
    ordered_cols = [col for col in expected_cols if col in df.columns]
    df = df[ordered_cols]

    output_filename = 'simulation_summary.csv'
    df.to_csv(output_filename, index=False, float_format='%.6f')
    print(f"分析完成，数据已保存到 '{output_filename}'。")

if __name__ == '__main__':
    main()