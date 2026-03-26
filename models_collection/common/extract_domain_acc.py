#!/usr/bin/env python3
"""
提取domain_test_acc中四个域的最大测试精度并生成CSV文件
"""
import json
import os
import re
import csv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_json_files(base_dir):
    """
    查找所有符合条件的JSON文件
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        list: (json_file_path, algorithm, prefix, train_names, test_names, embedding_rate) 的列表
    """
    json_files = []
    base_path = Path(base_dir)
    
    # 遍历所有算法目录
    for algorithm_dir in base_path.iterdir():
        if not algorithm_dir.is_dir() or algorithm_dir.name == 'common':
            continue
            
        algorithm = algorithm_dir.name
        
        # 查找所有train_*_to_*目录，可能在子目录中
        # 使用更宽泛的模式来匹配包含train_*_to_*的目录名
        for train_dir in algorithm_dir.rglob('*train_*_to_*'):
            if not train_dir.is_dir():
                continue
            
            # 提取路径前缀（如models/、temp/等）
            relative_path = train_dir.relative_to(algorithm_dir)
            path_prefix = str(relative_path.parent) if relative_path.parent != Path('.') else ""
            
            # 从目录名中提取prefix和train_names、test_names
            # 目录名格式可能是: {prefix}train_{train_names}_to_{test_names}
            dir_name = train_dir.name
            match = re.match(r'(.+?)train_(.+)_to_(.+)$', dir_name)
            if not match:
                # 如果格式不匹配，尝试不带prefix的格式
                match = re.match(r'train_(.+)_to_(.+)', dir_name)
                if not match:
                    continue
                prefix = ""
                train_names = match.group(1)
                test_names = match.group(2)
            else:
                prefix = match.group(1)  # 提取prefix（如sam_）
                train_names = match.group(2)
                test_names = match.group(3)
            
            # 合并路径prefix和目录名prefix
            if path_prefix:
                full_prefix = f"{path_prefix}/{prefix}" if prefix else path_prefix
            else:
                full_prefix = prefix
            
            # 查找匹配的JSON文件
            pattern = re.compile(r'train_logs_QIM\+PMS\+LSB\+AHCM_(.+)_1s\.json$')
            for json_file in train_dir.glob('train_logs_QIM+PMS+LSB+AHCM_*_1s.json'):
                match = pattern.match(json_file.name)
                if match:
                    embedding_rate = match.group(1)
                    json_files.append((str(json_file), algorithm, full_prefix, train_names, test_names, embedding_rate))
    
    return json_files


def extract_max_domain_acc(json_file_path):
    """
    从JSON文件中提取四个域的最大测试精度
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        dict: {'QIM': max_acc, 'PMS': max_acc, 'LSB': max_acc, 'AHCM': max_acc}
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        domain_test_acc = data.get('domain_test_acc', [])
        
        # 初始化最大值字典
        max_acc = {'QIM': 0.0, 'PMS': 0.0, 'LSB': 0.0, 'AHCM': 0.0}
        
        # 遍历所有epoch的domain_test_acc
        for epoch_data in domain_test_acc:
            if isinstance(epoch_data, dict) and epoch_data:
                for domain in ['QIM', 'PMS', 'LSB', 'AHCM']:
                    if domain in epoch_data:
                        acc_value = epoch_data[domain]
                        if isinstance(acc_value, (int, float)):
                            max_acc[domain] = max(max_acc[domain], acc_value)
        
        return max_acc
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return {'QIM': 0.0, 'PMS': 0.0, 'LSB': 0.0, 'AHCM': 0.0}


def generate_csv(results, output_dir):
    """
    生成CSV文件，按train_names和test_names分组
    
    Args:
        results: 结果列表，每个元素为 (algorithm, prefix, train_names, test_names, embedding_rate, max_acc_dict)
        output_dir: 输出目录
    """
    # 按train_names和test_names分组
    grouped = defaultdict(list)
    for result in results:
        algorithm, prefix, train_names, test_names, embedding_rate, max_acc = result
        key = (train_names, test_names)
        grouped[key].append((algorithm, prefix, embedding_rate, max_acc))
    
    # 为每个分组生成CSV文件
    for (train_names, test_names), items in grouped.items():
        # 生成文件名
        csv_filename = f"train_{train_names}_to_{test_names}_{items[0][2]}.csv"
        
        # 如果有多个embedding_rate，需要为每个生成一个文件
        embedding_rates = set(item[2] for item in items)
        for embedding_rate in embedding_rates:
            csv_filename = f"train_{train_names}_to_{test_names}_{embedding_rate}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # 获取该embedding_rate的所有项目
            rate_items = [item for item in items if item[2] == embedding_rate]
            
            # 写入CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Algorithm', 'QIM', 'PMS', 'LSB', 'AHCM'])
                
                for algorithm, prefix, _, max_acc in rate_items:
                    # 构建Algorithm名称：{prefix}algorithm
                    # prefix可能是目录名prefix（如sam_）或路径prefix（如models/）或两者结合
                    if prefix:
                        # 如果prefix包含斜杠，说明是路径prefix，需要保留
                        # 如果prefix是目录名prefix（如sam_），直接拼接
                        if '/' in prefix:
                            algorithm_name = f"{prefix}/{algorithm}"
                        else:
                            algorithm_name = f"{prefix}{algorithm}"
                    else:
                        algorithm_name = algorithm
                    
                    writer.writerow([
                        algorithm_name,
                        f"{max_acc['QIM']:.4f}",
                        f"{max_acc['PMS']:.4f}",
                        f"{max_acc['LSB']:.4f}",
                        f"{max_acc['AHCM']:.4f}"
                    ])


def main():
    base_dir = os.environ.get('DASM_MODELS_COLLECTION_ROOT', os.path.join(PROJECT_ROOT, 'models_collection'))
    output_dir = os.environ.get('DASM_EXTRACT_DOMAIN_ACC_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'models_collection', 'common'))
    
    print("正在查找JSON文件...")
    json_files = find_json_files(base_dir)
    print(f"找到 {len(json_files)} 个JSON文件")
    
    if not json_files:
        print("未找到符合条件的JSON文件")
        return
    
    results = []
    print("正在处理JSON文件...")
    for json_file, algorithm, prefix, train_names, test_names, embedding_rate in tqdm(json_files, desc="处理进度"):
        max_acc = extract_max_domain_acc(json_file)
        results.append((algorithm, prefix, train_names, test_names, embedding_rate, max_acc))
    
    print("正在生成CSV文件...")
    generate_csv(results, output_dir)
    print(f"CSV文件已生成到: {output_dir}")


if __name__ == '__main__':
    main()
