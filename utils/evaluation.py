"""
评估工具函数
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from scipy import stats as sp_stats

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HeartRateEvaluator:
    """
    多层级心率评估
    """
    
    def __init__(self, output_dir: str = './results'):
        """
        初始化评估器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def evaluate(self, results_df: pd.DataFrame) -> Dict:
        """
        执行多层级评估
        
        参数:
            results_df: 结果DataFrame
        
        返回:
            评估结果字典
        """
        print("\n" + "=" * 70)
        print("📊 心率估计评估结果")
        print("=" * 70)
        
        # 1. 片段级评估
        print("\n【片段级评估】")
        segment_metrics = self._calculate_metrics(
            results_df['radar_hr_bpm'].values,
            results_df['neulog_hr_interval'].values
        )
        self._print_metrics(segment_metrics, "所有片段")
        
        # 2. 文件级评估
        print("\n【文件级评估】")
        file_metrics = {}
        for file_id in sorted(results_df['file_unique_id'].unique()):
            file_data = results_df[results_df['file_unique_id'] == file_id]
            metrics = self._calculate_metrics(
                file_data['radar_hr_bpm'].values,
                file_data['neulog_hr_interval'].values
            )
            file_metrics[file_id] = metrics
            self._print_metrics(metrics, f"  {file_id}")
        
        # 3. 距离级评估
        print("\n【距离级评估】")
        distance_metrics = {}
        for distance in sorted(results_df['distance'].unique()):
            dist_data = results_df[results_df['distance'] == distance]
            metrics = self._calculate_metrics(
                dist_data['radar_hr_bpm'].values,
                dist_data['neulog_hr_interval'].values
            )
            distance_metrics[f"{distance}cm"] = metrics
            self._print_metrics(metrics, f"  {distance}cm")
        
        # 4. 生成可视化
        visualization_params = self._create_visualizations(results_df)
        
        # 5. 保存评估报告
        evaluation_results = {
            'segment': segment_metrics,
            'file': file_metrics,
            'distance': distance_metrics,
            'visualization': visualization_params
        }
        
        return evaluation_results
    
    def _calculate_metrics(self, pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        # 移除NaN值
        valid_mask = ~(np.isnan(pred) | np.isnan(true))
        pred = pred[valid_mask]
        true = true[valid_mask]
        
        if len(pred) == 0:
            return {
                'MAE': np.nan,
                'RMSE': np.nan,
                'Correlation': np.nan,
                'n_samples': 0
            }
        
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        corr = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else np.nan
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': corr,
            'n_samples': len(pred)
        }
    
    def _print_metrics(self, metrics: Dict[str, float], label: str):
        """打印评估指标"""
        print(f"{label}:")
        print(f"  样本数: {metrics['n_samples']}")
        if metrics['n_samples'] > 0:
            print(f"  MAE: {metrics['MAE']:.2f} bpm")
            print(f"  RMSE: {metrics['RMSE']:.2f} bpm")
            print(f"  相关系数: {metrics['Correlation']:.4f}")
    
    def _create_visualizations(self, results_df: pd.DataFrame) -> Dict:
        """生成可视化图表"""
        visualization_params = {}
        
        # 1. Bland-Altman图
        visualization_params['bland_altman'] = self._bland_altman_plot(results_df)
        
        # 2. 散点图 + 回归线
        visualization_params['scatter_regression'] = self._scatter_plot_with_regression(results_df)
        
        # 3. 按距离分组的箱线图
        visualization_params['boxplot_by_distance'] = self._boxplot_by_distance(results_df)
        
        # 4. 误差分布直方图
        visualization_params['error_distribution'] = self._error_distribution_plot(results_df)
        
        return visualization_params
    
    def _bland_altman_plot(self, results_df: pd.DataFrame) -> Dict:
        """Bland-Altman一致性分析图"""
        radar_hr = results_df['radar_hr_bpm'].values
        neulog_hr = results_df['neulog_hr_interval'].values
        
        valid_mask = ~(np.isnan(radar_hr) | np.isnan(neulog_hr))
        radar_hr = radar_hr[valid_mask]
        neulog_hr = neulog_hr[valid_mask]
        
        mean_hr = (radar_hr + neulog_hr) / 2
        diff_hr = radar_hr - neulog_hr
        
        mean_diff = np.mean(diff_hr)
        std_diff = np.std(diff_hr)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(mean_hr, diff_hr, alpha=0.6, s=50)
        ax.axhline(mean_diff, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean Diff: {mean_diff:.2f} bpm')
        ax.axhline(upper_loa, color='gray', linestyle='--', linewidth=1.5, 
                  label=f'+1.96 SD: {upper_loa:.2f} bpm')
        ax.axhline(lower_loa, color='gray', linestyle='--', linewidth=1.5, 
                  label=f'-1.96 SD: {lower_loa:.2f} bpm')
        
        ax.set_xlabel('Mean of Radar and NeuLog HR (bpm)', fontsize=12)
        ax.set_ylabel('Difference (Radar - NeuLog) (bpm)', fontsize=12)
        ax.set_title('Bland-Altman Plot: Radar vs NeuLog Heart Rate', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bland_altman_plot.png'), dpi=300)
        print(f"✅ Bland-Altman图已保存")
        plt.close()
        
        return {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'upper_loa': float(upper_loa),
            'lower_loa': float(lower_loa),
            'loa_range': float(upper_loa - lower_loa)
        }
    
    def _scatter_plot_with_regression(self, results_df: pd.DataFrame) -> Dict:
        """散点图 + 回归线"""
        radar_hr = results_df['radar_hr_bpm'].values
        neulog_hr = results_df['neulog_hr_interval'].values
        distances = results_df['distance'].values
        
        valid_mask = ~(np.isnan(radar_hr) | np.isnan(neulog_hr))
        radar_hr = radar_hr[valid_mask]
        neulog_hr = neulog_hr[valid_mask]
        distances = distances[valid_mask]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for dist in sorted(np.unique(distances)):
            mask = distances == dist
            ax.scatter(neulog_hr[mask], radar_hr[mask], 
                      label=f'{int(dist)}cm', alpha=0.6, s=60)
        
        min_hr = min(neulog_hr.min(), radar_hr.min())
        max_hr = max(neulog_hr.max(), radar_hr.max())
        ax.plot([min_hr, max_hr], [min_hr, max_hr], 'k--', linewidth=2, label='Ideal (y=x)')
        
        slope, intercept = np.polyfit(neulog_hr, radar_hr, 1)
        regression_line = slope * neulog_hr + intercept
        ax.plot(neulog_hr, regression_line, 'r-', linewidth=2, 
               label=f'Regression (y={slope:.2f}x+{intercept:.2f})')
        
        correlation = np.corrcoef(neulog_hr, radar_hr)[0, 1]
        r_squared = correlation ** 2
        predicted = slope * neulog_hr + intercept
        residuals = radar_hr - predicted
        residual_std = np.std(residuals)
        
        ax.set_xlabel('NeuLog Heart Rate (bpm)', fontsize=12)
        ax.set_ylabel('Radar Estimated Heart Rate (bpm)', fontsize=12)
        ax.set_title('Radar vs NeuLog Heart Rate Estimation', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scatter_with_regression.png'), dpi=300)
        print(f"✅ 散点图已保存")
        plt.close()
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'correlation': float(correlation),
            'residual_std': float(residual_std),
            'regression_equation': f'y = {slope:.4f}x + {intercept:.4f}'
        }
    
    def _boxplot_by_distance(self, results_df: pd.DataFrame) -> Dict:
        """按距离分组的误差箱线图"""
        results_df['error_bpm'] = results_df['radar_hr_bpm'] - results_df['neulog_hr_interval']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        distances = sorted(results_df['distance'].unique())
        data_to_plot = [results_df[results_df['distance'] == d]['error_bpm'].dropna().values 
                       for d in distances]
        
        ax.boxplot(data_to_plot, labels=[f'{int(d)}cm' for d in distances])
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Error (Radar - NeuLog) (bpm)', fontsize=12)
        ax.set_title('Heart Rate Estimation Error by Distance', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'boxplot_by_distance.png'), dpi=300)
        print(f"✅ 箱线图已保存")
        plt.close()
        
        boxplot_stats = {}
        for dist, data in zip(distances, data_to_plot):
            if len(data) > 0:
                boxplot_stats[f'{int(dist)}cm'] = {
                    'median': float(np.median(data)),
                    'q1': float(np.percentile(data, 25)),
                    'q3': float(np.percentile(data, 75)),
                    'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'n_samples': int(len(data))
                }
        
        return boxplot_stats
    
    def _error_distribution_plot(self, results_df: pd.DataFrame) -> Dict:
        """误差分布直方图"""
        errors = (results_df['radar_hr_bpm'] - results_df['neulog_hr_interval']).dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                  label=f'Mean Error: {errors.mean():.2f} bpm')
        
        ax.set_xlabel('Error (Radar - NeuLog) (bpm)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Heart Rate Estimation Errors', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'), dpi=300)
        print(f"✅ 误差分布图已保存")
        plt.close()
        
        mean_error = float(errors.mean())
        std_error = float(errors.std())
        median_error = float(errors.median())
        skewness = float(sp_stats.skew(errors))
        kurtosis = float(sp_stats.kurtosis(errors))
        
        percentiles = {
            '5th': float(np.percentile(errors, 5)),
            '25th': float(np.percentile(errors, 25)),
            '50th': float(np.percentile(errors, 50)),
            '75th': float(np.percentile(errors, 75)),
            '95th': float(np.percentile(errors, 95))
        }
        
        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'median_error': median_error,
            'min_error': float(errors.min()),
            'max_error': float(errors.max()),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'percentiles': percentiles,
            'error_range': float(errors.max() - errors.min())
        }