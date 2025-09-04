#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析PRICE/datas/workloads/pretrain/下所有workloads_cleaned.sql文件
将相同模板但参数不同的SQL查询分组保存为txt文件
参考query_files/TPCH的格式，按数据库名分组存放
"""

import re
import os
import glob
from collections import defaultdict
from typing import List, Dict, Tuple

def normalize_sql(sql: str) -> str:
    """标准化SQL查询，移除多余空格和换行"""
    # 移除多余空格和换行
    sql = re.sub(r'\s+', ' ', sql.strip())
    return sql

def extract_sql_template(sql: str) -> str:
    """提取SQL模板，将数值参数替换为占位符"""
    # 将数字替换为占位符
    template = re.sub(r'\b\d+\b', 'N', sql)
    # 将字符串字面量替换为占位符
    template = re.sub(r"'[^']*'", 'S', template)
    # 将双引号字符串替换为占位符
    template = re.sub(r'"[^"]*"', 'S', template)
    return template

def parse_workload_file(file_path: str) -> Dict[str, List[str]]:
    """解析单个workload文件，按SQL模板分组"""
    templates = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # 标准化SQL
            normalized_sql = normalize_sql(line)
            if not normalized_sql:
                continue
                
            # 提取模板
            template = extract_sql_template(normalized_sql)
            
            # 按模板分组
            templates[template].append(normalized_sql)
            
            # 每1000行打印进度
            if line_num % 1000 == 0:
                print(f"已处理 {line_num} 行...")
    
    return templates

def save_templates_to_files(templates: Dict[str, List[str]], output_dir: str, db_name: str):
    """将模板分组保存为txt文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 按查询数量排序，优先处理查询数量多的模板
    sorted_templates = sorted(templates.items(), key=lambda x: len(x[1]), reverse=True)
    
    saved_count = 0
    for i, (template, queries) in enumerate(sorted_templates, 1):
        if len(queries) < 2:  # 跳过只有一个查询的模板
            continue
            
        filename = f"{db_name.upper()}_{i}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for query in queries:
                f.write(f"  {query} \n")
        
        saved_count += 1
        print(f"保存 {filename}: {len(queries)} 个查询")
        print(f"模板: {template[:100]}...")
        print("-" * 80)
    
    return saved_count

def find_all_workload_files(base_dir: str) -> List[Tuple[str, str]]:
    """查找所有workloads_cleaned.sql文件"""
    pattern = os.path.join(base_dir, "*", "workloads_cleaned.sql")
    files = glob.glob(pattern)
    
    result = []
    for file_path in files:
        # 提取数据库名（文件夹名）
        db_name = os.path.basename(os.path.dirname(file_path))
        result.append((db_name, file_path))
    
    return result

def main():
    base_dir = "PRICE/datas/workloads/pretrain"
    output_base_dir = "query_files"
    
    print("开始查找所有workload文件...")
    workload_files = find_all_workload_files(base_dir)
    
    print(f"找到 {len(workload_files)} 个workload文件:")
    for db_name, file_path in workload_files:
        print(f"  {db_name}: {file_path}")
    
    print("\n" + "=" * 80)
    
    total_stats = {
        'total_databases': len(workload_files),
        'total_templates': 0,
        'total_queries': 0,
        'total_files_saved': 0
    }
    
    for db_name, file_path in workload_files:
        print(f"\n处理数据库: {db_name}")
        print(f"文件: {file_path}")
        print("-" * 60)
        
        try:
            # 解析文件
            templates = parse_workload_file(file_path)
            
            print(f"发现 {len(templates)} 个不同的SQL模板")
            
            # 统计信息
            total_queries = sum(len(queries) for queries in templates.values())
            multi_query_templates = sum(1 for queries in templates.values() if len(queries) >= 2)
            
            print(f"总查询数: {total_queries}")
            print(f"包含多个查询的模板数: {multi_query_templates}")
            
            # 保存文件
            output_dir = os.path.join(output_base_dir, db_name.upper())
            files_saved = save_templates_to_files(templates, output_dir, db_name)
            
            # 更新总统计
            total_stats['total_templates'] += len(templates)
            total_stats['total_queries'] += total_queries
            total_stats['total_files_saved'] += files_saved
            
            print(f"完成 {db_name}: 保存了 {files_saved} 个文件")
            
        except Exception as e:
            print(f"处理 {db_name} 时出错: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("处理完成！总统计:")
    print(f"处理的数据库数: {total_stats['total_databases']}")
    print(f"总模板数: {total_stats['total_templates']}")
    print(f"总查询数: {total_stats['total_queries']}")
    print(f"保存的文件数: {total_stats['total_files_saved']}")
    print(f"输出目录: {output_base_dir}")

if __name__ == "__main__":
    main() 