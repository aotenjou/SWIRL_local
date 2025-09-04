"""
SWIRL索引分析工具
用于分析和可视化SWIRL模型在测试时选择的索引
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import logging


class IndexAnalyzer:
    def __init__(self, experiment_folder_path):
        """
        初始化索引分析器
        
        Args:
            experiment_folder_path: 实验文件夹路径
        """
        self.experiment_folder_path = experiment_folder_path
        self.index_files = self._find_index_files()
        
    def _find_index_files(self):
        """查找索引选择文件"""
        index_files = []
        for file in os.listdir(self.experiment_folder_path):
            if file.startswith("index_selection_") and file.endswith(".json"):
                index_files.append(os.path.join(self.experiment_folder_path, file))
        return sorted(index_files)
    
    def load_index_data(self, file_path=None):
        """
        加载索引选择数据
        
        Args:
            file_path: 指定文件路径，如果为None则加载所有文件
            
        Returns:
            dict: 索引选择数据
        """
        if file_path:
            files = [file_path]
        else:
            files = self.index_files
            
        all_data = []
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.append(data)
            except Exception as e:
                logging.error(f"加载文件 {file} 时出错: {e}")
                
        return all_data
    
    def analyze_index_selection_patterns(self, data):
        """
        分析索引选择模式
        
        Args:
            data: 索引选择数据
            
        Returns:
            dict: 分析结果
        """
        analysis = {
            'total_episodes': 0,
            'total_indexes_selected': 0,
            'index_frequency': defaultdict(int),
            'step_distribution': defaultdict(int),
            'storage_usage': [],
            'cost_improvements': [],
            'index_sequences': []
        }
        
        for env_data in data:
            for episode in env_data['episodes']:
                analysis['total_episodes'] += 1
                analysis['total_indexes_selected'] += len(episode['final_indexes'])
                analysis['storage_usage'].append(episode['storage_consumption_mb'])
                analysis['cost_improvements'].append(episode['cost_improvement_percent'])
                
                # 统计索引频率
                for index in episode['final_indexes']:
                    analysis['index_frequency'][index] += 1
                
                # 统计步骤分布
                for seq_item in episode['index_selection_sequence']:
                    analysis['step_distribution'][seq_item['step']] += 1
                
                # 记录索引选择顺序
                analysis['index_sequences'].append({
                    'episode_id': episode['episode_id'],
                    'sequence': [item['index'] for item in episode['index_selection_sequence']]
                })
        
        return analysis
    
    def generate_summary_report(self, analysis):
        """
        生成摘要报告
        
        Args:
            analysis: 分析结果
            
        Returns:
            str: 摘要报告
        """
        report = []
        report.append("=" * 60)
        report.append("SWIRL索引选择分析报告")
        report.append("=" * 60)
        report.append(f"总工作负载数: {analysis['total_episodes']}")
        report.append(f"总选择索引数: {analysis['total_indexes_selected']}")
        report.append(f"平均每个工作负载索引数: {analysis['total_indexes_selected'] / analysis['total_episodes']:.2f}")
        report.append(f"平均存储消耗: {sum(analysis['storage_usage']) / len(analysis['storage_usage']):.2f} MB")
        report.append(f"平均成本改进: {sum(analysis['cost_improvements']) / len(analysis['cost_improvements']):.2f}%")
        
        report.append("\n最常选择的索引 (前10):")
        sorted_indexes = sorted(analysis['index_frequency'].items(), key=lambda x: x[1], reverse=True)
        for i, (index, freq) in enumerate(sorted_indexes[:10]):
            report.append(f"  {i+1}. {index}: {freq} 次")
        
        report.append("\n索引选择步骤分布:")
        sorted_steps = sorted(analysis['step_distribution'].items())
        for step, count in sorted_steps:
            report.append(f"  步骤 {step}: {count} 次")
        
        return "\n".join(report)
    
    def plot_index_frequency(self, analysis, top_n=20):
        """
        绘制索引频率图
        
        Args:
            analysis: 分析结果
            top_n: 显示前N个最常选择的索引
        """
        sorted_indexes = sorted(analysis['index_frequency'].items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        indexes = [item[0] for item in sorted_indexes]
        frequencies = [item[1] for item in sorted_indexes]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indexes)), frequencies)
        plt.yticks(range(len(indexes)), indexes)
        plt.xlabel('选择次数')
        plt.title(f'最常选择的索引 (前{top_n}个)')
        plt.tight_layout()
        
        output_path = os.path.join(self.experiment_folder_path, 'index_frequency.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"索引频率图已保存到: {output_path}")
    
    def plot_storage_vs_improvement(self, analysis):
        """
        绘制存储消耗vs成本改进散点图
        
        Args:
            analysis: 分析结果
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(analysis['storage_usage'], analysis['cost_improvements'], alpha=0.6)
        plt.xlabel('存储消耗 (MB)')
        plt.ylabel('成本改进 (%)')
        plt.title('存储消耗 vs 成本改进')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.experiment_folder_path, 'storage_vs_improvement.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"存储vs改进图已保存到: {output_path}")
    
    def plot_index_selection_timeline(self, analysis, max_episodes=10):
        """
        绘制索引选择时间线
        
        Args:
            analysis: 分析结果
            max_episodes: 最大显示的工作负载数
        """
        if not analysis['index_sequences']:
            print("没有找到索引选择序列数据")
            return
            
        fig, axes = plt.subplots(min(max_episodes, len(analysis['index_sequences'])), 1, 
                                figsize=(12, 3 * min(max_episodes, len(analysis['index_sequences']))))
        if max_episodes == 1:
            axes = [axes]
            
        for i, seq_data in enumerate(analysis['index_sequences'][:max_episodes]):
            sequence = seq_data['sequence']
            episode_id = seq_data['episode_id']
            
            # 创建时间线
            x_positions = list(range(1, len(sequence) + 1))
            
            axes[i].bar(x_positions, [1] * len(sequence), alpha=0.7)
            axes[i].set_ylabel(f'工作负载 {episode_id}')
            axes[i].set_xticks(x_positions)
            axes[i].set_xticklabels(sequence, rotation=45, ha='right')
            axes[i].set_ylim(0, 1.2)
            
            if i == len(analysis['index_sequences'][:max_episodes]) - 1:
                axes[i].set_xlabel('选择步骤')
        
        plt.suptitle('索引选择时间线')
        plt.tight_layout()
        
        output_path = os.path.join(self.experiment_folder_path, 'index_selection_timeline.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"索引选择时间线已保存到: {output_path}")
    
    def export_to_csv(self, analysis):
        """
        导出分析结果到CSV文件
        
        Args:
            analysis: 分析结果
        """
        # 导出索引频率
        freq_df = pd.DataFrame(list(analysis['index_frequency'].items()), 
                              columns=['Index', 'Frequency'])
        freq_df = freq_df.sort_values('Frequency', ascending=False)
        freq_path = os.path.join(self.experiment_folder_path, 'index_frequency.csv')
        freq_df.to_csv(freq_path, index=False)
        
        # 导出步骤分布
        step_df = pd.DataFrame(list(analysis['step_distribution'].items()),
                              columns=['Step', 'Count'])
        step_df = step_df.sort_values('Step')
        step_path = os.path.join(self.experiment_folder_path, 'step_distribution.csv')
        step_df.to_csv(step_path, index=False)
        
        # 导出存储和成本改进数据
        perf_df = pd.DataFrame({
            'Storage_MB': analysis['storage_usage'],
            'Cost_Improvement_Percent': analysis['cost_improvements']
        })
        perf_path = os.path.join(self.experiment_folder_path, 'performance_metrics.csv')
        perf_df.to_csv(perf_path, index=False)
        
        print(f"分析结果已导出到CSV文件:")
        print(f"  索引频率: {freq_path}")
        print(f"  步骤分布: {step_path}")
        print(f"  性能指标: {perf_path}")
    
    def run_full_analysis(self):
        """
        运行完整分析
        
        Returns:
            dict: 分析结果
        """
        print("开始分析SWIRL索引选择结果...")
        
        # 加载数据
        data = self.load_index_data()
        if not data:
            print("未找到索引选择数据文件")
            return None
            
        # 分析数据
        analysis = self.analyze_index_selection_patterns(data)
        
        # 生成报告
        report = self.generate_summary_report(analysis)
        print(report)
        
        # 保存报告到文件
        report_path = os.path.join(self.experiment_folder_path, 'index_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 生成图表
        try:
            self.plot_index_frequency(analysis)
            self.plot_storage_vs_improvement(analysis)
            self.plot_index_selection_timeline(analysis)
        except Exception as e:
            print(f"生成图表时出错: {e}")
        
        # 导出CSV
        self.export_to_csv(analysis)
        
        print(f"\n完整分析报告已保存到: {report_path}")
        
        return analysis


def main():
    """主函数，用于命令行调用"""
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python index_analyzer.py <experiment_folder_path>")
        sys.exit(1)
    
    experiment_path = sys.argv[1]
    if not os.path.exists(experiment_path):
        print(f"实验文件夹不存在: {experiment_path}")
        sys.exit(1)
    
    analyzer = IndexAnalyzer(experiment_path)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 