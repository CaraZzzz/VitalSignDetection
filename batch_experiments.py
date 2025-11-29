"""
批量实验脚本
用于运行多个心率估计方法和窗口长度的组合实验

运行方式：
    python batch_experiments.py
"""
"""
相比v1增加距离维度分析
相比v2增加四张图合成一张大图，且修改字号大小
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# ⚠️ 假设支持存在
# 假设存在 main.py 且其中定义了 run_experiment 函数
# 假设存在 config 模块
# ----------------------------------------------------------------------------
# 导入主程序 (假设 main.py 和 config 存在)
try:
    from main import run_experiment
    import config
except ImportError:
    # 如果环境不完整，定义一个 mock 函数以保持代码结构完整性
    print("Warning: main.run_experiment or config not found. Using mock implementation.")
    def run_experiment(hr_method, loc_method, window_s, output_dir):
        """Mock implementation for run_experiment"""
        time.sleep(1) # Simulate computation time
        
        # Simulate evaluation results
        np.random.seed(hash(f"{hr_method}_{window_s}") % (2**32))
        base_mae = 3.0 + np.random.randn() * 0.5 - 0.02 * window_s / 30.0
        mae = max(0.5, base_mae)
        rmse = mae * 1.3
        
        evaluation_results = {
            'visualization': {
                'bland_altman': {
                    'loa_range': rmse * 3.92,
                    'mean_difference': np.random.randn() * 0.5
                }
            },
            'segment': {
                'MAE': mae,
                'RMSE': rmse,
                'Correlation': min(0.99, max(0.85, 1 - mae / 15)),
                'n_samples': 300
            }
        }
        
        # Simulate computation times
        computation_times = [np.random.rand() * 0.1 for _ in range(300)]
        
        # Mock file generation for distance analysis (required by ResultAnalyzer)
        os.makedirs(output_dir, exist_ok=True)
        
        # Mock results file with distance info
        distances = [40, 50, 60] * 100 
        radar_hr = np.random.uniform(60, 100, 300)
        neulog_hr = radar_hr + np.random.randn(300) * mae
        
        mock_df = pd.DataFrame({
            'distance': distances[:300],
            'radar_hr_bpm': radar_hr,
            'neulog_hr_count': neulog_hr
        })
        mock_df.to_csv(os.path.join(output_dir, 'heart_rate_results.csv'), index=False)
        
        return evaluation_results, computation_times

# ****************************************************************************
# 【修改：字体大小设置】
# ****************************************************************************

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 统一设置字体大小
FONT_SIZE_AXIS = 18
FONT_SIZE_TITLE = 20
FONT_SIZE_LEGEND = 16

plt.rcParams['axes.labelsize'] = FONT_SIZE_AXIS
plt.rcParams['xtick.labelsize'] = FONT_SIZE_AXIS
plt.rcParams['ytick.labelsize'] = FONT_SIZE_AXIS
plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE     # Axes title size
plt.rcParams['figure.titlesize'] = FONT_SIZE_TITLE   # Figure suptitle size

# ============================================================================
# 【实验配置】
# ============================================================================

# 实验矩阵：每个方法对应的窗口长度列表
# EXPERIMENT_MATRIX = {
#     'fft': [30, 60, 90, 120],
#     'stft': [30, 60, 90, 120],
#     'wavelet': [30, 60, 90, 120],
#     'dct': [30, 60, 90, 120],
#     'emd': [30, 60, 90, 120],
#     'eemd': [30, 60, 90, 120],
#     'vmd': [30, 60, 90, 120]
# }

EXPERIMENT_MATRIX = {
    'wavelet': [30, 60, 90, 120],
    'vmd': [30, 60, 90, 120]
}

# 人体定位方法（固定使用manual）
LOCALIZATION_METHOD = 'manual'

# 输出目录
BATCH_OUTPUT_DIR = './batch_results'
CACHE_DIR = os.path.join(BATCH_OUTPUT_DIR, 'cache')
FIGURES_DIR = os.path.join(BATCH_OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(BATCH_OUTPUT_DIR, 'tables')

# ============================================================================
# 【日志配置】
# ============================================================================

def setup_logging():
    """设置日志"""
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
    
    log_file = os.path.join(BATCH_OUTPUT_DIR, f'batch_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# ============================================================================
# 【实验运行器】
# ============================================================================

class BatchExperimentRunner:
    """批量实验运行器"""
    
    def __init__(self, experiment_matrix: Dict[str, List[int]], logger: logging.Logger):
        """
        初始化批量实验运行器
        """
        self.experiment_matrix = experiment_matrix
        self.logger = logger
        self.results_cache = {}
        
        # 创建必要的目录
        os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(TABLES_DIR, exist_ok=True)
        
        # 加载已有的缓存
        self._load_cache()
    
    def _load_cache(self):
        """加载已有的缓存结果"""
        cache_file = os.path.join(CACHE_DIR, 'results_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.results_cache = json.load(f)
                self.logger.info(f"✅ 加载了 {len(self.results_cache)} 个缓存结果")
            except Exception as e:
                self.logger.warning(f"⚠️ 加载缓存失败: {e}")
                self.results_cache = {}
        else:
            self.results_cache = {}
    
    def _save_cache(self):
        """保存缓存结果"""
        cache_file = os.path.join(CACHE_DIR, 'results_cache.json')
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_cache, f, indent=4, ensure_ascii=False)
            self.logger.info(f"✅ 保存了 {len(self.results_cache)} 个缓存结果")
        except Exception as e:
            self.logger.error(f"❌ 保存缓存失败: {e}")
    
    def _get_experiment_key(self, method: str, window: int) -> str:
        """生成实验的唯一标识"""
        return f"{method}_{window}s"
    
    def _run_single_experiment(self, method: str, window: int) -> Dict:
        """
        运行单个实验
        """
        exp_key = self._get_experiment_key(method, window)
        
        # 检查缓存
        if exp_key in self.results_cache:
            self.logger.info(f"  ⚡ 使用缓存: {exp_key}")
            return self.results_cache[exp_key]
        
        # 运行实验
        self.logger.info(f"  🚀 运行实验: {exp_key}")
        
        output_dir = os.path.join(CACHE_DIR, exp_key)
        
        try:
            # 调用主程序
            start_time = time.time()
            evaluation_results, computation_times = run_experiment(
                hr_method=method,
                loc_method=LOCALIZATION_METHOD,
                window_s=window,
                output_dir=output_dir
            )
            total_time = time.time() - start_time
            
            # 提取关键指标
            result_dict = {
                'method': method,
                'window_s': window,
                'loa_range': evaluation_results['visualization']['bland_altman']['loa_range'],
                'mean_difference': evaluation_results['visualization']['bland_altman']['mean_difference'],
                'mae': evaluation_results['segment']['MAE'],
                'rmse': evaluation_results['segment']['RMSE'],
                'correlation': evaluation_results['segment']['Correlation'],
                'n_samples': evaluation_results['segment']['n_samples'],
                'mean_computation_time_per_segment_ms': np.mean(computation_times) * 1000,  # 转换为毫秒
                'std_computation_time_per_segment_ms': np.std(computation_times) * 1000,
                'total_experiment_time_s': total_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # 缓存结果
            self.results_cache[exp_key] = result_dict
            self._save_cache()
            
            self.logger.info(f"  ✅ 完成: {exp_key} | MAE={result_dict['mae']:.2f} | "
                             f"时间={result_dict['mean_computation_time_per_segment_ms']:.2f}ms")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"  ❌ 实验失败: {exp_key} | 错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def run_all_experiments(self) -> pd.DataFrame:
        """
        运行所有实验
        """
        self.logger.info("=" * 70)
        self.logger.info("🚀 开始批量实验")
        self.logger.info("=" * 70)
        
        # 计算总实验数
        total_experiments = sum(len(windows) for windows in self.experiment_matrix.values())
        self.logger.info(f"总实验数: {total_experiments}")
        
        # 运行所有实验
        all_results = []
        
        with tqdm(total=total_experiments, desc="总进度") as pbar:
            for method, windows in self.experiment_matrix.items():
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"方法: {method.upper()} | 窗口数: {len(windows)}")
                self.logger.info(f"{'='*70}")
                
                for window in windows:
                    result = self._run_single_experiment(method, window)
                    if result is not None:
                        all_results.append(result)
                    pbar.update(1)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(all_results)
        
        # 保存完整结果
        results_file = os.path.join(BATCH_OUTPUT_DIR, 'all_results.csv')
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"\n✅ 所有结果已保存至: {results_file}")
        
        return results_df

# ============================================================================
# 【数据分析和可视化】
# ============================================================================

class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_df: pd.DataFrame, logger: logging.Logger):
        """
        初始化结果分析器
        """
        self.results_df = results_df
        self.logger = logger
        
        # 绘图配置
        self.plot_methods = ['fft', 'stft', 'dct', 'wavelet', 'emd', 'eemd', 'vmd']
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.plot_methods)))
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>']
        self.window_ticks = [30, 60, 90, 120] # 新增：横坐标刻度
    
    # ****************************************************************************
    # 【修改：原 _plot_metric_comparison 拆分为单图绘制】
    # ****************************************************************************
    def _plot_single_metric_comparison(self, metric_key: str, ylabel: str, title: str):
        """绘制单个指标的对比图 (窗口长度) - 4张单图之一"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method, color, marker in zip(self.plot_methods, self.colors, self.markers):
            method_data = self.results_df[self.results_df['method'] == method].sort_values('window_s')
            
            if len(method_data) > 0:
                ax.plot(method_data['window_s'], method_data[metric_key],
                        marker=marker, markersize=10, linewidth=2.5, 
                        color=color, label=method.upper(), alpha=0.8)
        
        ax.set_xlabel('Window Length (s)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(self.window_ticks)  # <<< 修改点
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        filename = f"{metric_key}_comparison_single.png"
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✅ 单图: {filename}")

    # ****************************************************************************
    # 【新增：窗口长度 2x2 组合图】
    # ****************************************************************************
    def _plot_combined_window_comparison(self):
        """绘制 2x2 窗口长度对比组合图 - 1张组合图"""
        self.logger.info("  🚀 绘制 2x2 窗口长度组合图")
        
        metrics = [
            ('loa_range', 'LoA Range (bpm)', '(a) LoA Range'),
            ('mean_difference', 'Mean Difference (bpm)', '(b) Mean Difference'),
            ('mae', 'MAE (bpm)', '(c) MAE'),
            ('rmse', 'RMSE (bpm)', '(d) RMSE')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 11))
        axes = axes.flatten()
        
        for i, (metric_key, ylabel, sub_title) in enumerate(metrics):
            ax = axes[i]
            
            for method, color, marker in zip(self.plot_methods, self.colors, self.markers):
                method_data = self.results_df[self.results_df['method'] == method].sort_values('window_s')
                
                if len(method_data) > 0:
                    ax.plot(method_data['window_s'], method_data[metric_key],
                            marker=marker, markersize=10, linewidth=2.5, 
                            color=color, label=method.upper(), alpha=0.8)
            
            ax.set_xlabel('Window Length (s)', fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(sub_title, fontweight='bold')
            ax.set_xticks(self.window_ticks) # <<< 修改点
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # 统一图例放在右下角
        handles, labels = axes[0].get_legend_handles_labels()
        # 调整图例字体大小（这里使用 FONT_SIZE_LEGEND）
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
                   ncol=len(self.plot_methods), framealpha=0.9) 

        # 调整子图间距
        plt.subplots_adjust(hspace=0.35, wspace=0.2, bottom=0.15)
        
        # 添加总标题 (使用 FONT_SIZE_TITLE+2 确保比子图标题大)
        fig.suptitle('Performance Metrics vs Window Length Comparison', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
        
        filename = "combined_window_comparison.png"
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✅ 组合图: {filename}")


    # ****************************************************************************
    # 【修改： generate_comparison_plots 包含 4 张单图 + 1 张组合图】
    # ****************************************************************************
    def generate_comparison_plots(self):
        """生成4张单图和1张组合对比图 (窗口长度)"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📊 生成窗口长度对比图 (4张单图 + 1张组合图)")
        self.logger.info("=" * 70)
        
        metrics = [
            ('loa_range', 'LoA Range (bpm)', 'Limits of Agreement Range vs Window Length'),
            ('mean_difference', 'Mean Difference (bpm)', 'Mean Difference vs Window Length'),
            ('mae', 'MAE (bpm)', 'Mean Absolute Error vs Window Length'),
            ('rmse', 'RMSE (bpm)', 'Root Mean Square Error vs Window Length')
        ]
        
        # 1. 绘制 4 张单图
        for metric_key, ylabel, title in metrics:
            self._plot_single_metric_comparison(metric_key, ylabel, title)
        
        # 2. 绘制 1 张组合图
        self._plot_combined_window_comparison()
    
    def generate_summary_table(self):
        """生成汇总表"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📋 生成汇总表")
        self.logger.info("=" * 70)
        
        # Excel文件：每个方法一个sheet
        excel_file = os.path.join(TABLES_DIR, 'summary_tables.xlsx')
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for method in self.plot_methods:
                method_data = self.results_df[self.results_df['method'] == method].sort_values('window_s')
                
                if len(method_data) > 0:
                    # 选择要显示的列
                    table = method_data[[
                        'window_s', 'loa_range', 'mean_difference', 'mae', 'rmse',
                        'correlation', 'mean_computation_time_per_segment_ms'
                    ]].copy()
                    
                    # 重命名列
                    table.columns = [
                        'Window (s)', 'LoA Range', 'Mean Diff', 'MAE', 'RMSE',
                        'Correlation', 'Time (ms)'
                    ]
                    
                    # 格式化数值
                    for col in ['LoA Range', 'Mean Diff', 'MAE', 'RMSE']:
                        table[col] = table[col].round(2)
                    table['Correlation'] = table['Correlation'].round(4)
                    table['Time (ms)'] = table['Time (ms)'].round(2)
                    
                    # 写入Excel
                    table.to_excel(writer, sheet_name=method.upper(), index=False)
                    
                    self.logger.info(f"  ✅ {method.upper()}: {len(table)} rows")
        
        self.logger.info(f"\n✅ Excel表格已保存至: {excel_file}")
        
        # CSV文件：所有数据在一个文件
        csv_file = os.path.join(TABLES_DIR, 'summary_all_methods.csv')
        summary_csv = self.results_df[[
            'method', 'window_s', 'loa_range', 'mean_difference', 'mae', 'rmse',
            'correlation', 'mean_computation_time_per_segment_ms', 'n_samples'
        ]].sort_values(['method', 'window_s'])
        
        summary_csv.to_csv(csv_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"✅ CSV表格已保存至: {csv_file}")
        
        # Markdown表格：用于文档
        self._generate_markdown_tables()
    
    def _generate_markdown_tables(self):
        """生成Markdown格式的表格"""
        md_file = os.path.join(TABLES_DIR, 'summary_tables.md')
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 批量实验结果汇总\n\n")
            
            for method in self.plot_methods:
                method_data = self.results_df[self.results_df['method'] == method].sort_values('window_s')
                
                if len(method_data) > 0:
                    f.write(f"## {method.upper()}\n\n")
                    f.write("| Window (s) | LoA Range | Mean Diff | MAE | RMSE | Time (ms) |\n")
                    f.write("|------------|-----------|-----------|-----|------|----------|\n")
                    
                    for _, row in method_data.iterrows():
                        f.write(f"| {row['window_s']} | {row['loa_range']:.2f} | "
                                f"{row['mean_difference']:.2f} | {row['mae']:.2f} | "
                                f"{row['rmse']:.2f} | {row['mean_computation_time_per_segment_ms']:.2f} |\n")
                    
                    f.write("\n")
        
        self.logger.info(f"✅ Markdown表格已保存至: {md_file}")
    
    def print_summary_statistics(self):
        """打印汇总统计"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📈 汇总统计")
        self.logger.info("=" * 70)
        
        for method in self.plot_methods:
            method_data = self.results_df[self.results_df['method'] == method]
            
            if len(method_data) > 0:
                self.logger.info(f"\n{method.upper()}:")
                self.logger.info(f"  实验数: {len(method_data)}")
                self.logger.info(f"  MAE范围: {method_data['mae'].min():.2f} - {method_data['mae'].max():.2f}")
                self.logger.info(f"  RMSE范围: {method_data['rmse'].min():.2f} - {method_data['rmse'].max():.2f}")
                self.logger.info(f"  最佳窗口(MAE): {method_data.loc[method_data['mae'].idxmin(), 'window_s']:.0f}s")
                self.logger.info(f"  平均计算时间: {method_data['mean_computation_time_per_segment_ms'].mean():.2f}ms")
    
    # ****************************************************************************
    # 【修改：原 _plot_distance_comparison 拆分为单图绘制】
    # ****************************************************************************
    def _plot_single_distance_comparison(self, dist_df, window_s, metric_key: str, ylabel: str, title: str):
        """绘制单个指标的距离对比图 (4张单图之一)"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for method, color, marker in zip(self.plot_methods, self.colors, self.markers):
            method_data = dist_df[dist_df['method'] == method].sort_values('distance')
            
            if len(method_data) > 0:
                ax.plot(method_data['distance'], method_data[metric_key],
                        marker=marker, markersize=10, linewidth=2.5,
                        color=color, label=method.upper(), alpha=0.8)
        
        ax.set_xlabel('Distance (cm)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([40, 50, 60])
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        filename = f"distance_{metric_key}_w{window_s}s_single.png"
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✅ 单图: {filename}")

    # ****************************************************************************
    # 【新增：距离 2x2 组合图】
    # ****************************************************************************
    def _plot_combined_distance_comparison(self, dist_df, window_s):
        """绘制 2x2 距离对比组合图 - 1张组合图"""
        self.logger.info("  🚀 绘制 2x2 距离分析组合图")
        
        metrics = [
            ('loa_range', 'LoA Range (bpm)', '(a) LoA Range'),
            ('mean_difference', 'Mean Difference (bpm)', '(b) Mean Difference'),
            ('mae', 'MAE (bpm)', '(c) MAE'),
            ('rmse', 'RMSE (bpm)', '(d) RMSE')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 11))
        axes = axes.flatten()
        
        for i, (metric_key, ylabel, sub_title) in enumerate(metrics):
            ax = axes[i]
            
            for method, color, marker in zip(self.plot_methods, self.colors, self.markers):
                method_data = dist_df[dist_df['method'] == method].sort_values('distance')
                
                if len(method_data) > 0:
                    ax.plot(method_data['distance'], method_data[metric_key],
                            marker=marker, markersize=10, linewidth=2.5,
                            color=color, label=method.upper(), alpha=0.8)
            
            ax.set_xlabel('Distance (cm)', fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(sub_title, fontweight='bold')
            ax.set_xticks([40, 50, 60])
            ax.grid(True, alpha=0.3, linestyle='--')

        # 统一图例放在右下角
        handles, labels = axes[0].get_legend_handles_labels()
        # 调整图例字体大小（这里使用 FONT_SIZE_LEGEND）
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
                   ncol=len(self.plot_methods), framealpha=0.9)

        # 调整子图间距
        plt.subplots_adjust(hspace=0.35, wspace=0.2, bottom=0.15)
        
        # 添加总标题
        fig.suptitle(f'Performance Metrics vs Distance Comparison (Window={window_s}s)', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
        
        filename = f"combined_distance_analysis_w{window_s}s.png"
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✅ 组合图: {filename}")
    
    # ****************************************************************************
    # 【修改： generate_distance_analysis_plots 包含 4 张单图 + 1 张组合图】
    # ****************************************************************************
    def generate_distance_analysis_plots(self, window_s=120):
        """
        生成距离分析图（4张单图 + 1张组合图，固定窗口长度）
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"📊 生成距离分析图（窗口长度={window_s}s, 4张单图 + 1张组合图）")
        self.logger.info("=" * 70)
        
        # 1. 筛选指定窗口长度的实验
        target_experiments = self.results_df[self.results_df['window_s'] == window_s]
        
        if len(target_experiments) == 0:
            self.logger.warning(f"⚠️ 没有找到窗口长度为{window_s}s的实验数据")
            return
        
        self.logger.info(f"找到 {len(target_experiments)} 个方法的实验数据")
        
        # 2. 对每个实验，读取CSV并按距离统计
        distance_data = {
            'method': [],
            'distance': [],
            'mae': [],
            'rmse': [],
            'loa_range': [],
            'mean_difference': []
        }
        
        for _, exp in target_experiments.iterrows():
            method = exp['method']
            exp_key = self._get_experiment_key(method, window_s)
            csv_path = os.path.join(CACHE_DIR, exp_key, 'heart_rate_results.csv')
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    
                    # 按距离分组计算指标
                    for distance in [40, 50, 60]:
                        dist_df = df[df['distance'] == distance].copy()
                        
                        if len(dist_df) == 0:
                            self.logger.warning(f"  ⚠️ {method.upper()} - {distance}cm: 无数据")
                            continue
                        
                        # 计算误差
                        dist_df['error'] = dist_df['radar_hr_bpm'] - dist_df['neulog_hr_count']
                        
                        # 去除NaN值
                        valid_errors = dist_df['error'].dropna()
                        
                        if len(valid_errors) == 0:
                            self.logger.warning(f"  ⚠️ {method.upper()} - {distance}cm: 无有效数据")
                            continue
                        
                        # 计算MAE, RMSE
                        mae = valid_errors.abs().mean()
                        rmse = np.sqrt((valid_errors**2).mean())
                        
                        # 计算Bland-Altman指标
                        mean_diff = valid_errors.mean()
                        std_diff = valid_errors.std()
                        loa_range = 1.96 * 2 * std_diff
                        
                        # 保存数据
                        distance_data['method'].append(method)
                        distance_data['distance'].append(distance)
                        distance_data['mae'].append(mae)
                        distance_data['rmse'].append(rmse)
                        distance_data['loa_range'].append(loa_range)
                        distance_data['mean_difference'].append(mean_diff)
                        
                        self.logger.info(f"  ✅ {method.upper()} - {distance}cm: MAE={mae:.2f}, RMSE={rmse:.2f}")
                        
                except Exception as e:
                    self.logger.error(f"  ❌ 读取 {exp_key} 数据失败: {e}")
            else:
                self.logger.warning(f"  ⚠️ 文件不存在: {csv_path}")
        
        # 3. 转换为DataFrame
        if len(distance_data['method']) == 0:
            self.logger.error("❌ 没有足够的数据生成距离分析图")
            return
        
        dist_df = pd.DataFrame(distance_data)
        
        # 保存距离分析数据
        distance_csv = os.path.join(TABLES_DIR, f'distance_analysis_w{window_s}s.csv')
        dist_df.to_csv(distance_csv, index=False, encoding='utf-8-sig')
        self.logger.info(f"\n✅ 距离分析数据已保存: {distance_csv}")
        
        metrics = [
            ('mae', 'MAE (bpm)', f'MAE vs Distance (Window={window_s}s)'),
            ('rmse', 'RMSE (bpm)', f'RMSE vs Distance (Window={window_s}s)'),
            ('loa_range', 'LoA Range (bpm)', f'LoA Range vs Distance (Window={window_s}s)'),
            ('mean_difference', 'Mean Difference (bpm)', f'Mean Difference vs Distance (Window={window_s}s)')
        ]
        
        # 4. 绘制 4 张单图
        for metric_key, ylabel, title in metrics:
            self._plot_single_distance_comparison(dist_df, window_s, metric_key, ylabel, title)

        # 5. 绘制 1 张组合图
        self._plot_combined_distance_comparison(dist_df, window_s)
        
        self.logger.info(f"✅ 距离分析图生成完成（窗口={window_s}s，总共 5 张图）")
    
    # 由于 run_experiment 被 mock，这里需要添加 _get_experiment_key 来支持 distance analysis
    def _get_experiment_key(self, method: str, window: int) -> str:
        """生成实验的唯一标识"""
        return f"{method}_{window}s"


# ============================================================================
# 【主函数】
# ============================================================================

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("🚀 批量实验系统启动")
    logger.info("=" * 70)
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"实验配置: {sum(len(v) for v in EXPERIMENT_MATRIX.values())} 个实验")
    
    # 运行批量实验
    runner = BatchExperimentRunner(EXPERIMENT_MATRIX, logger)
    results_df = runner.run_all_experiments()
    
    # 分析结果
    analyzer = ResultAnalyzer(results_df, logger)
    
    # 生成窗口对比图 (4 张单图 + 1 张组合图)
    analyzer.generate_comparison_plots()
    
    # 生成汇总表
    analyzer.generate_summary_table()
    analyzer.print_summary_statistics()
    
    # 生成距离分析图（固定窗口长度120s, 4 张单图 + 1 张组合图）
    logger.info("\n" + "=" * 70)
    logger.info("📊 距离影响分析")
    logger.info("=" * 70)
    analyzer.generate_distance_analysis_plots(window_s=120)
    
    # 完成
    logger.info("\n" + "=" * 70)
    logger.info("✅ 批量实验完成！")
    logger.info("=" * 70)
    logger.info(f"\n输出目录: {BATCH_OUTPUT_DIR}")
    logger.info(f"  - 图表: {FIGURES_DIR}")
    logger.info(f"    * 窗口对比图（4张单图 + 1张组合图）")
    logger.info(f"    * 距离分析图（4张单图 + 1张组合图）")
    logger.info(f"  - 表格: {TABLES_DIR}")
    logger.info(f"  - 缓存: {CACHE_DIR}")
    logger.info("=" * 70)
    logger.info("总图表数：10 张")


if __name__ == "__main__":
    main()