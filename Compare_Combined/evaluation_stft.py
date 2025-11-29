"""
STFT双路对比评估工具
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


class STFTDualPathEvaluator:
    """
    STFT双路对比评估器
    分别评估使用预测Range Bin和真实Range Bin的心率估计效果
    """
    
    def __init__(self, output_dir: str = './results_stft'):
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
        执行双路评估
        
        参数:
            results_df: 包含所有预测结果的DataFrame
        
        返回:
            评估结果字典
        """
        print("\n" + "=" * 70)
        print("📊 STFT 心率估计双路对比评估")
        print("=" * 70)
        
        # 筛选错误预测的窗口
        wrong_df = results_df[results_df['is_correct'] == False].copy()
        
        if len(wrong_df) == 0:
            print("⚠️ 没有错误预测的窗口，无需进行STFT评估")
            return {}
        
        print(f"\n错误预测窗口总数: {len(wrong_df)}")
        
        # 1. 全局评估
        print("\n【全局评估】")
        pred_rb_metrics = self._calculate_metrics(
            wrong_df['hr_from_pred_rb'].values,
            wrong_df['neulog_hr_bpm'].values,
            label="使用预测Range Bin"
        )
        
        true_rb_metrics = self._calculate_metrics(
            wrong_df['hr_from_true_rb'].values,
            wrong_df['neulog_hr_bpm'].values,
            label="使用真实Range Bin"
        )
        
        # 2. 按距离评估
        print("\n【按距离评估】")
        distance_metrics = {}
        for distance in sorted(wrong_df['distance'].unique()):
            dist_data = wrong_df[wrong_df['distance'] == distance]
            print(f"\n  {int(distance)}cm ({len(dist_data)} 个错误窗口):")
            
            pred_metrics = self._calculate_metrics(
                dist_data['hr_from_pred_rb'].values,
                dist_data['neulog_hr_bpm'].values,
                label=f"    预测RB"
            )
            
            true_metrics = self._calculate_metrics(
                dist_data['hr_from_true_rb'].values,
                dist_data['neulog_hr_bpm'].values,
                label=f"    真实RB"
            )
            
            distance_metrics[f"{int(distance)}cm"] = {
                'pred_rb': pred_metrics,
                'true_rb': true_metrics
            }
        
        # 3. 生成可视化
        print("\n【生成可视化】")
        
        # 为预测RB生成图表
        pred_rb_dir = os.path.join(self.output_dir, 'pred_rb_evaluation')
        os.makedirs(pred_rb_dir, exist_ok=True)
        pred_viz = self._create_visualizations(
            wrong_df, 
            hr_column='hr_from_pred_rb',
            output_dir=pred_rb_dir,
            title_prefix='预测Range Bin'
        )
        
        # 为真实RB生成图表
        true_rb_dir = os.path.join(self.output_dir, 'true_rb_evaluation')
        os.makedirs(true_rb_dir, exist_ok=True)
        true_viz = self._create_visualizations(
            wrong_df, 
            hr_column='hr_from_true_rb',
            output_dir=true_rb_dir,
            title_prefix='真实Range Bin'
        )
        
        # 4. 保存评估结果
        evaluation_results = {
            'global': {
                'pred_rb': pred_rb_metrics,
                'true_rb': true_rb_metrics
            },
            'by_distance': distance_metrics,
            'visualization': {
                'pred_rb': pred_viz,
                'true_rb': true_viz
            }
        }
        
        # 保存JSON
        with open(os.path.join(self.output_dir, 'stft_evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        print(f"\n✅ 评估完成，结果已保存至: {self.output_dir}")
        
        return evaluation_results
    
    def _calculate_metrics(self, pred: np.ndarray, true: np.ndarray, label: str = "") -> Dict[str, float]:
        """计算评估指标"""
        # 移除NaN值
        valid_mask = ~(np.isnan(pred) | np.isnan(true))
        pred = pred[valid_mask]
        true = true[valid_mask]
        
        if len(pred) == 0:
            print(f"{label}: 无有效样本")
            return {
                'MAE': np.nan,
                'RMSE': np.nan,
                'Correlation': np.nan,
                'n_samples': 0
            }
        
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        corr = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else np.nan
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': corr,
            'n_samples': len(pred)
        }
        
        # 打印
        if label:
            print(f"{label}:")
            print(f"  样本数: {metrics['n_samples']}")
            print(f"  MAE: {metrics['MAE']:.2f} bpm")
            print(f"  RMSE: {metrics['RMSE']:.2f} bpm")
            print(f"  相关系数: {metrics['Correlation']:.4f}")
        
        return metrics
    
    def _create_visualizations(self, data_df: pd.DataFrame, hr_column: str, 
                              output_dir: str, title_prefix: str) -> Dict:
        """生成可视化图表"""
        
        visualization_params = {}
        
        # 1. Bland-Altman图
        visualization_params['bland_altman'] = self._bland_altman_plot(
            data_df, hr_column, output_dir, title_prefix
        )
        
        # 2. 散点图 + 回归线
        visualization_params['scatter_regression'] = self._scatter_plot_with_regression(
            data_df, hr_column, output_dir, title_prefix
        )
        
        # 3. 按距离分组的箱线图
        visualization_params['boxplot_by_distance'] = self._boxplot_by_distance(
            data_df, hr_column, output_dir, title_prefix
        )
        
        # 4. 误差分布直方图
        visualization_params['error_distribution'] = self._error_distribution_plot(
            data_df, hr_column, output_dir, title_prefix
        )
        
        return visualization_params
    
    def _bland_altman_plot(self, data_df: pd.DataFrame, hr_column: str, 
                          output_dir: str, title_prefix: str) -> Dict:
        """Bland-Altman一致性分析图"""
        radar_hr = data_df[hr_column].values
        neulog_hr = data_df['neulog_hr_bpm'].values
        
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
        ax.set_title(f'Bland-Altman Plot: {title_prefix}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bland_altman.png'), dpi=300)
        plt.close()
        
        return {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'upper_loa': float(upper_loa),
            'lower_loa': float(lower_loa)
        }
    
    def _scatter_plot_with_regression(self, data_df: pd.DataFrame, hr_column: str,
                                     output_dir: str, title_prefix: str) -> Dict:
        """散点图 + 回归线"""
        radar_hr = data_df[hr_column].values
        neulog_hr = data_df['neulog_hr_bpm'].values
        distances = data_df['distance'].values
        
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
        
        ax.set_xlabel('NeuLog Heart Rate (bpm)', fontsize=12)
        ax.set_ylabel('Radar Estimated Heart Rate (bpm)', fontsize=12)
        ax.set_title(f'Scatter Plot: {title_prefix}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scatter_regression.png'), dpi=300)
        plt.close()
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'correlation': float(correlation)
        }
    
    def _boxplot_by_distance(self, data_df: pd.DataFrame, hr_column: str,
                            output_dir: str, title_prefix: str) -> Dict:
        """按距离分组的误差箱线图"""
        data_df = data_df.copy()
        data_df['error_bpm'] = data_df[hr_column] - data_df['neulog_hr_bpm']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        distances = sorted(data_df['distance'].unique())
        data_to_plot = [data_df[data_df['distance'] == d]['error_bpm'].dropna().values 
                       for d in distances]
        
        ax.boxplot(data_to_plot, labels=[f'{int(d)}cm' for d in distances])
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Error (Radar - NeuLog) (bpm)', fontsize=12)
        ax.set_title(f'Error by Distance: {title_prefix}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boxplot_by_distance.png'), dpi=300)
        plt.close()
        
        boxplot_stats = {}
        for dist, data in zip(distances, data_to_plot):
            if len(data) > 0:
                boxplot_stats[f'{int(dist)}cm'] = {
                    'median': float(np.median(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'n_samples': int(len(data))
                }
        
        return boxplot_stats
    
    def _error_distribution_plot(self, data_df: pd.DataFrame, hr_column: str,
                                output_dir: str, title_prefix: str) -> Dict:
        """误差分布直方图"""
        errors = (data_df[hr_column] - data_df['neulog_hr_bpm']).dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                  label=f'Mean Error: {errors.mean():.2f} bpm')
        
        ax.set_xlabel('Error (Radar - NeuLog) (bpm)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Error Distribution: {title_prefix}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
        plt.close()
        
        return {
            'mean_error': float(errors.mean()),
            'std_error': float(errors.std()),
            'median_error': float(errors.median()),
            'min_error': float(errors.min()),
            'max_error': float(errors.max())
        }