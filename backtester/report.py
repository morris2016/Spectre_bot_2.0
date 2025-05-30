#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Backtester Report Module

This module generates comprehensive reports and visualizations for backtest results,
including performance metrics, trade analysis, and strategy insights.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import base64
from io import BytesIO
import tempfile
import jinja2
import pdfkit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from common.exceptions import ReportGenerationError
from common.utils import create_unique_id, safe_divide

logger = logging.getLogger(__name__)


class BacktestReport:
    """
    Generates comprehensive reports for backtest results with deep analysis
    and visualizations.
    """

    def __init__(self, 
                report_output_dir: str = None,
                template_dir: str = None):
        """Initialize the backtest report generator."""
        self.report_output_dir = report_output_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'reports')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.report_output_dir):
            os.makedirs(self.report_output_dir)
            
        # Set up template environment
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), 'templates')
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Set Matplotlib and Seaborn styles
        plt.style.use('dark_background')
        sns.set_style("darkgrid")
        sns.set_context("talk")
        
        # Color schemes
        self.colors = {
            'profit': '#4CAF50',
            'loss': '#F44336',
            'equity': '#2196F3',
            'drawdown': '#FF9800',
            'buy': '#26a69a',
            'sell': '#ef5350',
            'background': '#121212',
            'text': '#e0e0e0',
            'grid': '#333333'
        }
        
        logger.info(f"Initialized backtest report generator with output dir: {self.report_output_dir}")

    def generate_report(self, 
                       backtest_result: Dict[str, Any],
                       performance_metrics: Dict[str, Any],
                       strategy_info: Dict[str, Any],
                       output_format: str = 'html',
                       filename: Optional[str] = None) -> str:
        """
        Generate a comprehensive backtest report.
        
        Args:
            backtest_result: The raw backtest result data
            performance_metrics: Calculated performance metrics
            strategy_info: Information about the strategy
            output_format: 'html', 'pdf', or 'json'
            filename: Optional filename for the report
            
        Returns:
            str: Path to the generated report file
        """
        try:
            # Generate a report ID and filename if not provided
            report_id = create_unique_id()
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                strategy_name = strategy_info.get('name', 'unknown').replace(' ', '_')
                filename = f"{strategy_name}_{timestamp}_{report_id[:8]}"
            
            # Create report data structure
            report_data = self._prepare_report_data(
                backtest_result, performance_metrics, strategy_info)
            
            # Generate the report in requested format
            if output_format == 'html':
                return self._generate_html_report(report_data, filename)
            elif output_format == 'pdf':
                return self._generate_pdf_report(report_data, filename)
            elif output_format == 'json':
                return self._generate_json_report(report_data, filename)
            else:
                raise ReportGenerationError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
            raise ReportGenerationError(f"Failed to generate report: {e}")

    def _prepare_report_data(self, 
                            backtest_result: Dict[str, Any],
                            performance_metrics: Dict[str, Any],
                            strategy_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data structure for the report, including generating all visualizations.
        
        Args:
            backtest_result: Raw backtest results
            performance_metrics: Calculated performance metrics
            strategy_info: Strategy information
            
        Returns:
            Dict containing all data needed for the report
        """
        # Extract key information
        trades = backtest_result.get('trades', [])
        equity_curve = backtest_result.get('equity_curve', pd.DataFrame())
        market_data = backtest_result.get('market_data', pd.DataFrame())
        
        # Process trades data
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Generate visualizations as base64 encoded images
        visualizations = {}
        
        # Only generate visualizations if we have data
        if not trades_df.empty and not equity_curve.empty:
            visualizations['equity_curve'] = self._generate_equity_curve_plot(equity_curve)
            visualizations['drawdown_chart'] = self._generate_drawdown_chart(equity_curve)
            visualizations['monthly_returns'] = self._generate_monthly_returns_heatmap(equity_curve)
            visualizations['trade_distribution'] = self._generate_trade_distribution_plot(trades_df)
            visualizations['profit_loss_histogram'] = self._generate_profit_loss_histogram(trades_df)
            visualizations['win_loss_distribution'] = self._generate_win_loss_distribution(trades_df)
            
            if not market_data.empty:
                visualizations['trades_on_price'] = self._generate_trades_on_price_chart(
                    market_data, trades_df)
                    
            visualizations['trade_durations'] = self._generate_trade_duration_analysis(trades_df)
            visualizations['cumulative_trades'] = self._generate_cumulative_trades_chart(trades_df)
            
        # Generate trade analysis
        trade_analysis = self._analyze_trades(trades_df)
        
        # Generate strategy insights
        strategy_insights = self._generate_strategy_insights(
            trades_df, performance_metrics, strategy_info)
        
        # Prepare summary metrics
        summary_metrics = self._prepare_summary_metrics(performance_metrics)
        
        # Assemble final report data
        report_data = {
            'report_id': create_unique_id(),
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy_info': strategy_info,
            'summary_metrics': summary_metrics,
            'performance_metrics': performance_metrics,
            'trade_analysis': trade_analysis,
            'strategy_insights': strategy_insights,
            'visualizations': visualizations,
            'trades': trades_df.to_dict('records') if not trades_df.empty else [],
            'backtest_params': backtest_result.get('params', {})
        }
        
        return report_data

    def _generate_html_report(self, report_data: Dict[str, Any], 
                             filename: str) -> str:
        """Generate an HTML report from the prepared data."""
        try:
            # Load the HTML template
            template = self.jinja_env.get_template('backtest_report.html')
            
            # Render the template with our data
            html_content = template.render(**report_data)
            
            # Save the HTML file
            file_path = os.path.join(self.report_output_dir, f"{filename}.html")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Generated HTML report: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise ReportGenerationError(f"Failed to generate HTML report: {e}")

    def _generate_pdf_report(self, report_data: Dict[str, Any], 
                            filename: str) -> str:
        """Generate a PDF report from the prepared data."""
        try:
            # First generate HTML
            html_path = self._generate_html_report(report_data, f"{filename}_temp")
            
            # Convert to PDF
            pdf_path = os.path.join(self.report_output_dir, f"{filename}.pdf")
            
            # Use pdfkit to convert HTML to PDF
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': 'UTF-8',
                'no-outline': None
            }
            
            pdfkit.from_file(html_path, pdf_path, options=options)
            
            # Remove temporary HTML file
            os.remove(html_path)
            
            logger.info(f"Generated PDF report: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise ReportGenerationError(f"Failed to generate PDF report: {e}")

    def _generate_json_report(self, report_data: Dict[str, Any], 
                             filename: str) -> str:
        """Generate a JSON report from the prepared data."""
        try:
            # Create a copy of report data with visualizations as paths instead of base64
            json_data = report_data.copy()
            
            # Save visualizations as separate image files
            if 'visualizations' in json_data:
                viz_dir = os.path.join(self.report_output_dir, f"{filename}_viz")
                if not os.path.exists(viz_dir):
                    os.makedirs(viz_dir)
                
                viz_paths = {}
                for viz_name, viz_data in json_data['visualizations'].items():
                    # Skip if not base64 data
                    if not viz_data or not viz_data.startswith('data:image'):
                        continue
                    
                    # Save image to file
                    img_format = 'png'
                    if 'data:image/jpeg' in viz_data:
                        img_format = 'jpeg'
                    
                    img_data = viz_data.split(',')[1]
                    img_path = os.path.join(viz_dir, f"{viz_name}.{img_format}")
                    
                    with open(img_path, 'wb') as f:
                        f.write(base64.b64decode(img_data))
                    
                    # Store relative path
                    viz_paths[viz_name] = os.path.relpath(img_path, self.report_output_dir)
                
                # Replace base64 data with paths
                json_data['visualizations'] = viz_paths
            
            # Save JSON file
            file_path = os.path.join(self.report_output_dir, f"{filename}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
                
            logger.info(f"Generated JSON report: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            raise ReportGenerationError(f"Failed to generate JSON report: {e}")

    def _prepare_summary_metrics(self, 
                                performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare summary metrics for the report dashboard."""
        summary = {
            'total_return': performance_metrics.get('total_return', 0),
            'total_return_formatted': f"{performance_metrics.get('total_return', 0):.2f}%",
            'win_rate': performance_metrics.get('win_rate', 0),
            'win_rate_formatted': f"{performance_metrics.get('win_rate', 0):.2f}%",
            'profit_factor': performance_metrics.get('profit_factor', 0),
            'profit_factor_formatted': f"{performance_metrics.get('profit_factor', 0):.2f}",
            'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
            'sharpe_ratio_formatted': f"{performance_metrics.get('sharpe_ratio', 0):.2f}",
            'max_drawdown': performance_metrics.get('max_drawdown', 0),
            'max_drawdown_formatted': f"{performance_metrics.get('max_drawdown', 0):.2f}%",
            'total_trades': performance_metrics.get('total_trades', 0),
            'winning_trades': performance_metrics.get('winning_trades', 0),
            'losing_trades': performance_metrics.get('losing_trades', 0),
            'avg_trade': performance_metrics.get('avg_trade', 0),
            'avg_trade_formatted': f"{performance_metrics.get('avg_trade', 0):.2f}%",
            'avg_win': performance_metrics.get('avg_win', 0),
            'avg_win_formatted': f"{performance_metrics.get('avg_win', 0):.2f}%",
            'avg_loss': performance_metrics.get('avg_loss', 0),
            'avg_loss_formatted': f"{performance_metrics.get('avg_loss', 0):.2f}%",
            'largest_win': performance_metrics.get('largest_win', 0),
            'largest_win_formatted': f"{performance_metrics.get('largest_win', 0):.2f}%",
            'largest_loss': performance_metrics.get('largest_loss', 0),
            'largest_loss_formatted': f"{performance_metrics.get('largest_loss', 0):.2f}%",
        }
        
        # Calculate additional metrics
        if summary['win_rate'] > 0 and 'avg_win' in performance_metrics and 'avg_loss' in performance_metrics:
            # Risk-reward ratio (absolute value of win/loss)
            if summary['avg_loss'] != 0:
                risk_reward = abs(summary['avg_win'] / summary['avg_loss'])
                summary['risk_reward_ratio'] = risk_reward
                summary['risk_reward_ratio_formatted'] = f"{risk_reward:.2f}"
            else:
                summary['risk_reward_ratio'] = float('inf')
                summary['risk_reward_ratio_formatted'] = "∞"
                
            # Expected value per trade
            expected_value = (summary['win_rate'] / 100 * summary['avg_win']) + \
                            ((100 - summary['win_rate']) / 100 * summary['avg_loss'])
            summary['expected_value'] = expected_value
            summary['expected_value_formatted'] = f"{expected_value:.2f}%"
            
            # Kelly criterion percentage
            if summary['avg_loss'] != 0:
                w = summary['win_rate'] / 100
                r = abs(summary['avg_win'] / summary['avg_loss'])
                kelly = (w * r - (1 - w)) / r
                kelly = max(0, min(kelly, 1)) * 100  # Bound between 0-100%
                summary['kelly_criterion'] = kelly
                summary['kelly_criterion_formatted'] = f"{kelly:.2f}%"
            else:
                summary['kelly_criterion'] = 100
                summary['kelly_criterion_formatted'] = "100.00%"
        
        return summary

    def _analyze_trades(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform detailed analysis on trades data."""
        if trades_df.empty:
            return {
                'error': 'No trades data available for analysis'
            }
        
        try:
            # Ensure we have necessary columns
            required_cols = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 
                           'direction', 'profit_pct', 'profit_amount']
            
            missing_cols = [col for col in required_cols if col not in trades_df.columns]
            if missing_cols:
                return {
                    'error': f"Missing required columns: {', '.join(missing_cols)}"
                }
            
            # Calculate trade durations if not already present
            if 'duration' not in trades_df.columns:
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time'])
                trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
            
            # Separate winning and losing trades
            winning_trades = trades_df[trades_df['profit_pct'] > 0]
            losing_trades = trades_df[trades_df['profit_pct'] <= 0]
            
            # Trade direction analysis
            long_trades = trades_df[trades_df['direction'] == 'long']
            short_trades = trades_df[trades_df['direction'] == 'short']
            
            long_wins = long_trades[long_trades['profit_pct'] > 0]
            long_losses = long_trades[long_trades['profit_pct'] <= 0]
            
            short_wins = short_trades[short_trades['profit_pct'] > 0]
            short_losses = short_trades[short_trades['profit_pct'] <= 0]
            
            # Time-based analysis
            trades_df['entry_hour'] = trades_df['entry_time'].dt.hour
            trades_df['entry_day'] = trades_df['entry_time'].dt.day_name()
            trades_df['entry_month'] = trades_df['entry_time'].dt.month
            
            # Hour performance
            hour_performance = trades_df.groupby('entry_hour')['profit_pct'].agg(
                ['mean', 'count', 'sum']).reset_index()
            hour_performance.columns = ['hour', 'avg_profit_pct', 'trade_count', 'total_profit_pct']
            
            best_hour = hour_performance.loc[hour_performance['avg_profit_pct'].idxmax()]
            worst_hour = hour_performance.loc[hour_performance['avg_profit_pct'].idxmin()]
            
            # Day performance
            day_performance = trades_df.groupby('entry_day')['profit_pct'].agg(
                ['mean', 'count', 'sum']).reset_index()
            day_performance.columns = ['day', 'avg_profit_pct', 'trade_count', 'total_profit_pct']
            
            # Order days correctly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_performance['day_order'] = day_performance['day'].map(
                {day: i for i, day in enumerate(day_order)})
            day_performance = day_performance.sort_values('day_order').drop('day_order', axis=1)
            
            best_day = day_performance.loc[day_performance['avg_profit_pct'].idxmax()]
            worst_day = day_performance.loc[day_performance['avg_profit_pct'].idxmin()]
            
            # Month performance
            month_performance = trades_df.groupby('entry_month')['profit_pct'].agg(
                ['mean', 'count', 'sum']).reset_index()
            month_performance.columns = ['month', 'avg_profit_pct', 'trade_count', 'total_profit_pct']
            
            # Consecutive wins/losses
            trades_df['win'] = trades_df['profit_pct'] > 0
            
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_consecutive_wins = 0
            current_consecutive_losses = 0
            
            for win in trades_df['win']:
                if win:
                    current_consecutive_wins += 1
                    current_consecutive_losses = 0
                else:
                    current_consecutive_losses += 1
                    current_consecutive_wins = 0
                
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            
            # Duration analysis
            avg_duration_all = trades_df['duration_hours'].mean()
            avg_duration_wins = winning_trades['duration_hours'].mean() if not winning_trades.empty else 0
            avg_duration_losses = losing_trades['duration_hours'].mean() if not losing_trades.empty else 0
            
            # Profit per hour of trade
            trades_df['profit_per_hour'] = trades_df['profit_pct'] / trades_df['duration_hours']
            trades_df['profit_per_hour'].replace([np.inf, -np.inf], np.nan, inplace=True)
            avg_profit_per_hour = trades_df['profit_per_hour'].mean()
            
            # Calculate streaks (runs of consecutive wins/losses)
            runs = []
            current_run = {'type': None, 'count': 0, 'profit_sum': 0}
            
            for i, row in trades_df.iterrows():
                is_win = row['profit_pct'] > 0
                
                if current_run['type'] is None:
                    # First trade in the backtest
                    current_run['type'] = 'win' if is_win else 'loss'
                    current_run['count'] = 1
                    current_run['profit_sum'] = row['profit_pct']
                elif (is_win and current_run['type'] == 'win') or (not is_win and current_run['type'] == 'loss'):
                    # Continuing the streak
                    current_run['count'] += 1
                    current_run['profit_sum'] += row['profit_pct']
                else:
                    # End of streak, start new one
                    runs.append(current_run.copy())
                    current_run['type'] = 'win' if is_win else 'loss'
                    current_run['count'] = 1
                    current_run['profit_sum'] = row['profit_pct']
            
            # Add the last run
            if current_run['count'] > 0:
                runs.append(current_run.copy())
            
            # Analyze runs
            win_runs = [run for run in runs if run['type'] == 'win']
            loss_runs = [run for run in runs if run['type'] == 'loss']
            
            avg_win_streak = np.mean([run['count'] for run in win_runs]) if win_runs else 0
            avg_loss_streak = np.mean([run['count'] for run in loss_runs]) if loss_runs else 0
            
            max_win_streak = max([run['count'] for run in win_runs]) if win_runs else 0
            max_loss_streak = max([run['count'] for run in loss_runs]) if loss_runs else 0
            
            # Assemble the analysis results
            analysis = {
                'trade_count': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0,
                
                'long_trades': {
                    'count': len(long_trades),
                    'wins': len(long_wins),
                    'losses': len(long_losses),
                    'win_rate': (len(long_wins) / len(long_trades) * 100) if len(long_trades) > 0 else 0,
                    'avg_profit': long_trades['profit_pct'].mean() if not long_trades.empty else 0
                },
                
                'short_trades': {
                    'count': len(short_trades),
                    'wins': len(short_wins),
                    'losses': len(short_losses),
                    'win_rate': (len(short_wins) / len(short_trades) * 100) if len(short_trades) > 0 else 0,
                    'avg_profit': short_trades['profit_pct'].mean() if not short_trades.empty else 0
                },
                
                'time_analysis': {
                    'best_hour': {
                        'hour': int(best_hour['hour']),
                        'avg_profit': float(best_hour['avg_profit_pct']),
                        'trade_count': int(best_hour['trade_count'])
                    } if not hour_performance.empty else {},
                    
                    'worst_hour': {
                        'hour': int(worst_hour['hour']),
                        'avg_profit': float(worst_hour['avg_profit_pct']),
                        'trade_count': int(worst_hour['trade_count'])
                    } if not hour_performance.empty else {},
                    
                    'best_day': {
                        'day': best_day['day'],
                        'avg_profit': float(best_day['avg_profit_pct']),
                        'trade_count': int(best_day['trade_count'])
                    } if not day_performance.empty else {},
                    
                    'worst_day': {
                        'day': worst_day['day'],
                        'avg_profit': float(worst_day['avg_profit_pct']),
                        'trade_count': int(worst_day['trade_count'])
                    } if not day_performance.empty else {},
                    
                    'hour_performance': hour_performance.to_dict('records'),
                    'day_performance': day_performance.to_dict('records'),
                    'month_performance': month_performance.to_dict('records')
                },
                
                'duration_analysis': {
                    'avg_duration_all_hours': avg_duration_all,
                    'avg_duration_wins_hours': avg_duration_wins,
                    'avg_duration_losses_hours': avg_duration_losses,
                    'avg_profit_per_hour': avg_profit_per_hour
                },
                
                'streak_analysis': {
                    'max_consecutive_wins': max_consecutive_wins,
                    'max_consecutive_losses': max_consecutive_losses,
                    'avg_win_streak': avg_win_streak,
                    'avg_loss_streak': avg_loss_streak,
                    'ratio_avg_win_to_loss_streak': avg_win_streak / avg_loss_streak if avg_loss_streak > 0 else float('inf')
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {
                'error': f"Failed to analyze trades: {e}"
            }

    def _generate_strategy_insights(self,
                                  trades_df: pd.DataFrame,
                                  performance_metrics: Dict[str, Any],
                                  strategy_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights and recommendations based on backtest results."""
        insights = {
            'strengths': [],
            'weaknesses': [],
            'opportunities': [],
            'recommendations': []
        }
        
        # Skip if no trades
        if trades_df.empty:
            insights['weaknesses'].append("No trades executed during the backtest period.")
            insights['recommendations'].append("Verify strategy entry conditions and adjust parameters to ensure triggers are generated in the test period.")
            return insights
        
        try:
            # Extract key metrics
            win_rate = performance_metrics.get('win_rate', 0)
            profit_factor = performance_metrics.get('profit_factor', 0)
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            total_return = performance_metrics.get('total_return', 0)
            avg_trade = performance_metrics.get('avg_trade', 0)
            avg_win = performance_metrics.get('avg_win', 0)
            avg_loss = performance_metrics.get('avg_loss', 0)
            total_trades = performance_metrics.get('total_trades', 0)
            
            # Risk-reward ratio
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Analyze strategy strengths
            if win_rate >= 60:
                insights['strengths'].append(f"High win rate of {win_rate:.2f}%, indicating good trade selection.")
                
            if profit_factor >= 2:
                insights['strengths'].append(f"Strong profit factor of {profit_factor:.2f}, showing good profitability relative to risk.")
                
            if risk_reward >= 1.5:
                insights['strengths'].append(f"Favorable risk-reward ratio of {risk_reward:.2f}, winning trades outperform losing trades.")
                
            if max_drawdown <= 15:
                insights['strengths'].append(f"Controlled maximum drawdown of {max_drawdown:.2f}%, indicating good risk management.")
                
            if sharpe_ratio >= 1.5:
                insights['strengths'].append(f"Strong risk-adjusted returns with Sharpe ratio of {sharpe_ratio:.2f}.")
                
            # Analyze strategy weaknesses
            if win_rate < 40:
                insights['weaknesses'].append(f"Low win rate of {win_rate:.2f}% may indicate poor entry/exit criteria.")
                
            if profit_factor < 1.2:
                insights['weaknesses'].append(f"Marginal profit factor of {profit_factor:.2f} suggests minimal edge over trading costs.")
                
            if risk_reward < 1:
                insights['weaknesses'].append(f"Poor risk-reward ratio of {risk_reward:.2f}, average losses exceed average wins.")
                
            if max_drawdown > 30:
                insights['weaknesses'].append(f"High maximum drawdown of {max_drawdown:.2f}% indicates significant risk exposure.")
                
            if sharpe_ratio < 0.5:
                insights['weaknesses'].append(f"Low Sharpe ratio of {sharpe_ratio:.2f} suggests poor risk-adjusted returns.")
                
            if total_trades < 30:
                insights['weaknesses'].append(f"Limited sample size of {total_trades} trades may not be statistically significant.")
            
            # Identify opportunities
            if win_rate >= 50 and risk_reward < 1.5:
                insights['opportunities'].append("Potential to improve overall returns by increasing position size on winning trades.")
                
            if win_rate < 50 and risk_reward > 1.5:
                insights['opportunities'].append("Strategy has good risk-reward ratio but needs improved entry criteria to increase win rate.")
                
            if max_drawdown > 20:
                insights['opportunities'].append("Implementing more aggressive stop-loss mechanisms could reduce drawdowns.")
                
            if 'long_trades' in strategy_info and 'short_trades' in strategy_info:
                # If we have direction-specific data
                long_win_rate = strategy_info['long_trades'].get('win_rate', 0)
                short_win_rate = strategy_info['short_trades'].get('win_rate', 0)
                
                if abs(long_win_rate - short_win_rate) > 15:
                    better_direction = "long" if long_win_rate > short_win_rate else "short"
                    insights['opportunities'].append(f"Strategy performs significantly better on {better_direction} trades. Consider specializing or adjusting parameters for the weaker direction.")
            
            # Generate recommendations
            if win_rate < 45:
                insights['recommendations'].append("Review entry criteria to improve trade selection accuracy.")
                
            if risk_reward < 1:
                insights['recommendations'].append("Adjust exit strategies to let profits run longer and/or tighten stop losses.")
                
            if profit_factor < 1.2:
                insights['recommendations'].append("Reconsider the strategy's edge in the chosen market conditions.")
                
            if max_drawdown > 25:
                insights['recommendations'].append("Implement stronger risk management controls to limit drawdowns.")
                
            if avg_trade > 0 and total_return > 0 and sharpe_ratio > 1:
                position_sizing = "more aggressive" if max_drawdown < 15 else "current"
                insights['recommendations'].append(f"Strategy shows positive expectancy. Consider {position_sizing} position sizing for optimal returns.")
                
            # Generate specific insights based on trade analysis
            time_insights = self._generate_time_based_insights(trades_df)
            if time_insights:
                insights['opportunities'].extend(time_insights)
            
            # Ensure we have at least some insights in each category
            default_categories = {
                'strengths': "Strategy executed according to its defined parameters.",
                'weaknesses': "Strategy may need further optimization for current market conditions.",
                'opportunities': "Further testing with parameter variations may yield improved results.",
                'recommendations': "Continue refining the strategy based on these backtest results."
            }
            
            for category, default_msg in default_categories.items():
                if not insights[category]:
                    insights[category].append(default_msg)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating strategy insights: {e}")
            return {
                'strengths': ["Strategy executed according to its defined parameters."],
                'weaknesses': ["Unable to generate detailed insights due to an error."],
                'opportunities': ["Further testing with parameter variations may yield improved results."],
                'recommendations': ["Review the strategy and backtest data for completeness."]
            }

    def _generate_time_based_insights(self, trades_df: pd.DataFrame) -> List[str]:
        """Generate insights based on time patterns in trading performance."""
        if trades_df.empty:
            return []
        
        insights = []
        
        try:
            # Convert time columns if needed
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            
            # Time of day analysis
            trades_df['hour'] = trades_df['entry_time'].dt.hour
            hour_performance = trades_df.groupby('hour')['profit_pct'].mean()
            
            best_hours = hour_performance.nlargest(3).index.tolist()
            worst_hours = hour_performance.nsmallest(3).index.tolist()
            
            if max(hour_performance) > 0 and hour_performance.max() > abs(hour_performance.min()) * 1.5:
                best_hours_str = ', '.join([f"{h}:00" for h in best_hours])
                insights.append(f"Strategy performs best during hours: {best_hours_str}. Consider focusing trading during these periods.")
            
            if min(hour_performance) < 0 and abs(hour_performance.min()) > hour_performance.max() * 1.5:
                worst_hours_str = ', '.join([f"{h}:00" for h in worst_hours])
                insights.append(f"Strategy performs poorly during hours: {worst_hours_str}. Consider avoiding trades during these periods.")
            
            # Day of week analysis
            trades_df['day_of_week'] = trades_df['entry_time'].dt.day_name()
            day_performance = trades_df.groupby('day_of_week')['profit_pct'].mean()
            
            best_day = day_performance.idxmax()
            worst_day = day_performance.idxmin()
            
            if day_performance[best_day] > 0 and day_performance[best_day] > abs(day_performance[worst_day]) * 1.5:
                insights.append(f"Strategy shows strongest performance on {best_day}. Consider increasing position sizes on this day.")
            
            if day_performance[worst_day] < 0 and abs(day_performance[worst_day]) > day_performance[best_day] * 1.5:
                insights.append(f"Strategy consistently underperforms on {worst_day}. Consider avoiding trades or reducing position sizes on this day.")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating time-based insights: {e}")
            return []

    def _generate_equity_curve_plot(self, equity_curve: pd.DataFrame) -> str:
        """Generate equity curve visualization."""
        try:
            if equity_curve.empty:
                return ""

            # Create figure
            plt.figure(figsize=(12, 6), facecolor=self.colors['background'])
            
            # Plot equity curve
            plt.plot(equity_curve.index, equity_curve['equity'], 
                    color=self.colors['equity'], linewidth=2,
                    label='Equity Curve')
            
            if 'benchmark' in equity_curve.columns:
                # Plot benchmark if available
                plt.plot(equity_curve.index, equity_curve['benchmark'], 
                        color='#9C27B0', linewidth=1.5, linestyle='--',
                        label='Benchmark')
            
            # Configure plot
            plt.title('Strategy Equity Curve', color=self.colors['text'], fontsize=16)
            plt.xlabel('Date', color=self.colors['text'])
            plt.ylabel('Equity', color=self.colors['text'])
            plt.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5)
            plt.legend()
            
            # Style adjustments
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_color(self.colors['grid'])
            plt.gca().spines['left'].set_color(self.colors['grid'])
            plt.xticks(color=self.colors['text'])
            plt.yticks(color=self.colors['text'])
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating equity curve plot: {e}")
            return ""

    def _generate_drawdown_chart(self, equity_curve: pd.DataFrame) -> str:
        """Generate drawdown visualization."""
        try:
            if equity_curve.empty or 'drawdown_pct' not in equity_curve.columns:
                return ""
            
            # Create figure
            plt.figure(figsize=(12, 6), facecolor=self.colors['background'])
            
            # Plot drawdown
            plt.fill_between(equity_curve.index, 0, -equity_curve['drawdown_pct'], 
                           color=self.colors['drawdown'], alpha=0.6)
            plt.plot(equity_curve.index, -equity_curve['drawdown_pct'], 
                    color=self.colors['drawdown'], linewidth=1)
            
            # Configure plot
            plt.title('Drawdown Chart', color=self.colors['text'], fontsize=16)
            plt.xlabel('Date', color=self.colors['text'])
            plt.ylabel('Drawdown (%)', color=self.colors['text'])
            plt.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5)
            
            # Set Y-axis to display positive percentages
            plt.gca().invert_yaxis()
            
            # Style adjustments
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_color(self.colors['grid'])
            plt.gca().spines['left'].set_color(self.colors['grid'])
            plt.xticks(color=self.colors['text'])
            plt.yticks(color=self.colors['text'])
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating drawdown chart: {e}")
            return ""

    def _generate_monthly_returns_heatmap(self, equity_curve: pd.DataFrame) -> str:
        """Generate monthly returns heatmap."""
        try:
            if equity_curve.empty or 'returns' not in equity_curve.columns:
                return ""
            
            # Calculate monthly returns
            equity_curve = equity_curve.copy()
            equity_curve.index = pd.to_datetime(equity_curve.index)
            monthly_returns = equity_curve['returns'].resample('M').sum() * 100
            
            # Create a DataFrame with months as rows and years as columns
            returns_by_month = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack(0)
            
            # Convert month numbers to names
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            returns_by_month.index = [month_names[i-1] for i in returns_by_month.index]
            
            # Create figure
            plt.figure(figsize=(12, 8), facecolor=self.colors['background'])
            
            # Create custom colormap (red for negative, green for positive)
            from matplotlib.colors import LinearSegmentedColormap
            colors = [(0.8, 0.2, 0.2), (0.1, 0.1, 0.1), (0.2, 0.8, 0.2)]  # red, dark gray, green
            cmap = LinearSegmentedColormap.from_list('custom_RdGn', colors, N=100)
            
            # Plot heatmap
            ax = sns.heatmap(returns_by_month, annot=True, cmap=cmap, center=0,
                          fmt=".2f", linewidths=.5, cbar_kws={"label": "Return (%)"},
                          annot_kws={"size": 9, "weight": "bold"})
            
            # Configure plot
            plt.title('Monthly Returns (%)', color=self.colors['text'], fontsize=16)
            plt.xlabel('Year', color=self.colors['text'])
            plt.ylabel('Month', color=self.colors['text'])
            
            # Style adjustments
            plt.xticks(color=self.colors['text'])
            plt.yticks(color=self.colors['text'], rotation=0)
            ax.collections[0].colorbar.ax.yaxis.label.set_color(self.colors['text'])
            ax.collections[0].colorbar.ax.tick_params(colors=self.colors['text'])
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating monthly returns heatmap: {e}")
            return ""

    def _generate_trade_distribution_plot(self, trades_df: pd.DataFrame) -> str:
        """Generate trade distribution visualization."""
        try:
            if trades_df.empty or 'profit_pct' not in trades_df.columns:
                return ""
            
            # Create figure
            plt.figure(figsize=(12, 6), facecolor=self.colors['background'])
            
            # Separate winning and losing trades
            winning_trades = trades_df[trades_df['profit_pct'] > 0]['profit_pct']
            losing_trades = trades_df[trades_df['profit_pct'] <= 0]['profit_pct']
            
            # Plot histogram
            plt.hist(winning_trades, bins=20, color=self.colors['profit'], alpha=0.7, label='Winning Trades')
            plt.hist(losing_trades, bins=20, color=self.colors['loss'], alpha=0.7, label='Losing Trades')
            
            # Configure plot
            plt.title('Trade Profit/Loss Distribution', color=self.colors['text'], fontsize=16)
            plt.xlabel('Profit/Loss (%)', color=self.colors['text'])
            plt.ylabel('Number of Trades', color=self.colors['text'])
            plt.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5)
            plt.legend()
            
            # Style adjustments
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_color(self.colors['grid'])
            plt.gca().spines['left'].set_color(self.colors['grid'])
            plt.xticks(color=self.colors['text'])
            plt.yticks(color=self.colors['text'])
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating trade distribution plot: {e}")
            return ""

    def _generate_profit_loss_histogram(self, trades_df: pd.DataFrame) -> str:
        """Generate profit/loss histogram."""
        try:
            if trades_df.empty or 'profit_pct' not in trades_df.columns:
                return ""
            
            # Create figure
            plt.figure(figsize=(12, 6), facecolor=self.colors['background'])
            
            # Create bar chart of profits by trade
            trade_idx = range(len(trades_df))
            colors = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in trades_df['profit_pct']]
            
            plt.bar(trade_idx, trades_df['profit_pct'], color=colors, alpha=0.7)
            
            # Add horizontal line at 0
            plt.axhline(y=0, color=self.colors['grid'], linestyle='-', linewidth=1)
            
            # Configure plot
            plt.title('Profit/Loss by Trade', color=self.colors['text'], fontsize=16)
            plt.xlabel('Trade Number', color=self.colors['text'])
            plt.ylabel('Profit/Loss (%)', color=self.colors['text'])
            plt.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Style adjustments
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_color(self.colors['grid'])
            plt.gca().spines['left'].set_color(self.colors['grid'])
            plt.xticks(color=self.colors['text'])
            plt.yticks(color=self.colors['text'])
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating profit/loss histogram: {e}")
            return ""

    def _generate_win_loss_distribution(self, trades_df: pd.DataFrame) -> str:
        """Generate win/loss distribution chart."""
        try:
            if trades_df.empty or 'profit_pct' not in trades_df.columns:
                return ""
            
            # Calculate win/loss stats
            winning_trades = trades_df[trades_df['profit_pct'] > 0]
            losing_trades = trades_df[trades_df['profit_pct'] <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            total_count = len(trades_df)
            
            win_percentage = win_count / total_count * 100 if total_count > 0 else 0
            loss_percentage = loss_count / total_count * 100 if total_count > 0 else 0
            
            avg_win = winning_trades['profit_pct'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['profit_pct'].mean() if not losing_trades.empty else 0
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=self.colors['background'])
            
            # First subplot - Win/Loss count
            labels = ['Winning Trades', 'Losing Trades']
            sizes = [win_count, loss_count]
            colors = [self.colors['profit'], self.colors['loss']]
            explode = (0.1, 0)  # explode the 1st slice (Winning Trades)
            
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90, textprops={'color': self.colors['text']})
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax1.set_title('Win/Loss Distribution', color=self.colors['text'], fontsize=14)
            
            # Second subplot - Average Win/Loss
            labels = ['Avg Win (%)', 'Avg Loss (%)']
            values = [avg_win, abs(avg_loss)]  # Use absolute value for loss to show properly in bar chart
            bar_colors = [self.colors['profit'], self.colors['loss']]
            
            bars = ax2.bar(labels, values, color=bar_colors, alpha=0.7)
            
            # Add labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color=self.colors['text'], fontweight='bold')
            
            ax2.set_ylabel('Percentage (%)', color=self.colors['text'])
            ax2.set_title('Average Win vs. Loss', color=self.colors['text'], fontsize=14)
            ax2.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5, alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_color(self.colors['grid'])
            ax2.spines['left'].set_color(self.colors['grid'])
            ax2.tick_params(axis='x', colors=self.colors['text'])
            ax2.tick_params(axis='y', colors=self.colors['text'])
            
            plt.tight_layout()
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating win/loss distribution: {e}")
            return ""

    def _generate_trades_on_price_chart(self, market_data: pd.DataFrame, 
                                      trades_df: pd.DataFrame) -> str:
        """Generate a price chart with trade entry/exit points."""
        try:
            if market_data.empty or trades_df.empty:
                return ""
            
            # Ensure datetime index
            market_data = market_data.copy()
            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data.index = pd.to_datetime(market_data.index)
            
            # Convert trade times to datetime
            trades_df = trades_df.copy()
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Create figure
            plt.figure(figsize=(14, 8), facecolor=self.colors['background'])
            
            # Plot price chart
            plt.plot(market_data.index, market_data['close'], color='white', linewidth=1.5, label='Price')
            
            # Plot entry and exit points
            for _, trade in trades_df.iterrows():
                # Find entry and exit times in market data
                entry_idx = market_data.index.get_indexer([trade['entry_time']], method='nearest')[0]
                exit_idx = market_data.index.get_indexer([trade['exit_time']], method='nearest')[0]
                
                # Plot entry point
                marker_color = self.colors['buy'] if trade['direction'] == 'long' else self.colors['sell']
                plt.scatter(market_data.index[entry_idx], market_data['close'].iloc[entry_idx],
                          marker='^' if trade['direction'] == 'long' else 'v',
                          s=100, color=marker_color, edgecolors='white', zorder=5)
                
                # Plot exit point with profit/loss color
                exit_color = self.colors['profit'] if trade['profit_pct'] > 0 else self.colors['loss']
                plt.scatter(market_data.index[exit_idx], market_data['close'].iloc[exit_idx],
                          marker='o', s=80, color=exit_color, edgecolors='white', zorder=5)
            
            # Configure plot
            plt.title('Price Chart with Trades', color=self.colors['text'], fontsize=16)
            plt.xlabel('Date', color=self.colors['text'])
            plt.ylabel('Price', color=self.colors['text'])
            plt.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5)
            
            # Add legend
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color=self.colors['buy'], marker='^', linestyle='None', 
                      markersize=10, markeredgecolor='white'),
                Line2D([0], [0], color=self.colors['sell'], marker='v', linestyle='None', 
                      markersize=10, markeredgecolor='white'),
                Line2D([0], [0], color=self.colors['profit'], marker='o', linestyle='None', 
                      markersize=10, markeredgecolor='white'),
                Line2D([0], [0], color=self.colors['loss'], marker='o', linestyle='None', 
                      markersize=10, markeredgecolor='white')
            ]
            
            plt.legend(custom_lines, ['Long Entry', 'Short Entry', 'Profitable Exit', 'Loss Exit'],
                     loc='upper left', frameon=True, facecolor='black', edgecolor=self.colors['grid'])
            
            # Style adjustments
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_color(self.colors['grid'])
            plt.gca().spines['left'].set_color(self.colors['grid'])
            plt.xticks(color=self.colors['text'])
            plt.yticks(color=self.colors['text'])
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating trades on price chart: {e}")
            return ""

    def _generate_trade_duration_analysis(self, trades_df: pd.DataFrame) -> str:
        """Generate trade duration analysis visualization."""
        try:
            if trades_df.empty:
                return ""
            
            # Calculate trade durations if needed
            trades_df = trades_df.copy()
            
            if 'duration' not in trades_df.columns:
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time'])
            
            # Convert to hours for analysis
            if 'duration_hours' not in trades_df.columns:
                trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
            
            # Separate winning and losing trades
            winning_trades = trades_df[trades_df['profit_pct'] > 0]
            losing_trades = trades_df[trades_df['profit_pct'] <= 0]
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=self.colors['background'])
            
            # First subplot - Duration distribution
            max_duration = min(trades_df['duration_hours'].max() * 1.1, 500)  # Cap for extreme outliers
            bins = np.linspace(0, max_duration, 30)
            
            ax1.hist(winning_trades['duration_hours'], bins=bins, color=self.colors['profit'], 
                    alpha=0.7, label='Winning Trades')
            ax1.hist(losing_trades['duration_hours'], bins=bins, color=self.colors['loss'], 
                    alpha=0.7, label='Losing Trades')
            
            ax1.set_xlabel('Duration (Hours)', color=self.colors['text'])
            ax1.set_ylabel('Number of Trades', color=self.colors['text'])
            ax1.set_title('Trade Duration Distribution', color=self.colors['text'], fontsize=14)
            ax1.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5, alpha=0.3)
            ax1.legend()
            
            # Style adjustments for first subplot
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_color(self.colors['grid'])
            ax1.spines['left'].set_color(self.colors['grid'])
            ax1.tick_params(axis='x', colors=self.colors['text'])
            ax1.tick_params(axis='y', colors=self.colors['text'])
            
            # Second subplot - Duration vs. Profit correlation
            ax2.scatter(winning_trades['duration_hours'], winning_trades['profit_pct'], 
                       color=self.colors['profit'], alpha=0.7, label='Winning Trades')
            ax2.scatter(losing_trades['duration_hours'], losing_trades['profit_pct'], 
                       color=self.colors['loss'], alpha=0.7, label='Losing Trades')
            
            # Add trend line
            if len(trades_df) > 1:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    trades_df['duration_hours'], trades_df['profit_pct'])
                
                x = np.array([0, max_duration])
                y = intercept + slope * x
                ax2.plot(x, y, color='white', linestyle='--', linewidth=1)
                
                # Add correlation coefficient
                corr = trades_df['duration_hours'].corr(trades_df['profit_pct'])
                correlation_text = f'Correlation: {corr:.2f}'
                ax2.annotate(correlation_text, xy=(0.05, 0.95), xycoords='axes fraction',
                           color=self.colors['text'], fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
            
            ax2.set_xlabel('Duration (Hours)', color=self.colors['text'])
            ax2.set_ylabel('Profit/Loss (%)', color=self.colors['text'])
            ax2.set_title('Duration vs. Profit Correlation', color=self.colors['text'], fontsize=14)
            ax2.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5, alpha=0.3)
            ax2.axhline(y=0, color=self.colors['grid'], linestyle='-', linewidth=1)
            
            # Style adjustments for second subplot
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_color(self.colors['grid'])
            ax2.spines['left'].set_color(self.colors['grid'])
            ax2.tick_params(axis='x', colors=self.colors['text'])
            ax2.tick_params(axis='y', colors=self.colors['text'])
            
            plt.tight_layout()
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating trade duration analysis: {e}")
            return ""

    def _generate_cumulative_trades_chart(self, trades_df: pd.DataFrame) -> str:
        """Generate cumulative trades performance chart."""
        try:
            if trades_df.empty or 'profit_pct' not in trades_df.columns:
                return ""
            
            # Sort trades by entry time
            trades_df = trades_df.sort_values('entry_time').copy()
            
            # Calculate cumulative profit
            trades_df['cumulative_profit_pct'] = trades_df['profit_pct'].cumsum()
            
            # Create figure
            plt.figure(figsize=(12, 6), facecolor=self.colors['background'])
            
            # Plot cumulative profit
            plt.plot(range(len(trades_df)), trades_df['cumulative_profit_pct'], 
                    color=self.colors['equity'], linewidth=2)
            
            # Add markers for individual trades
            for i, (_, trade) in enumerate(trades_df.iterrows()):
                marker_color = self.colors['profit'] if trade['profit_pct'] > 0 else self.colors['loss']
                plt.scatter(i, trade['cumulative_profit_pct'], 
                          color=marker_color, s=20, zorder=5)
            
            # Configure plot
            plt.title('Cumulative Profit/Loss by Trade', color=self.colors['text'], fontsize=16)
            plt.xlabel('Trade Number', color=self.colors['text'])
            plt.ylabel('Cumulative Profit/Loss (%)', color=self.colors['text'])
            plt.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5)
            
            # Add horizontal line at 0
            plt.axhline(y=0, color=self.colors['grid'], linestyle='-', linewidth=1)
            
            # Style adjustments
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_color(self.colors['grid'])
            plt.gca().spines['left'].set_color(self.colors['grid'])
            plt.xticks(color=self.colors['text'])
            plt.yticks(color=self.colors['text'])
            
            # Get the image as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor=self.colors['background'])
            plt.close()
            
            # Convert to base64 string
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating cumulative trades chart: {e}")
            return ""


class InteractiveReport:
    """
    Generates interactive HTML reports with Plotly visualizations
    for deeper analysis and interactivity.
    """
    
    def __init__(self, 
                report_output_dir: str = None,
                template_dir: str = None):
        """Initialize the interactive report generator."""
        self.report_output_dir = report_output_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'reports', 'interactive')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.report_output_dir):
            os.makedirs(self.report_output_dir)
            
        # Set up template environment
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), 'templates')
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Color schemes
        self.colors = {
            'profit': '#4CAF50',
            'loss': '#F44336',
            'equity': '#2196F3',
            'drawdown': '#FF9800',
            'buy': '#26a69a',
            'sell': '#ef5350',
            'background': '#121212',
            'text': '#e0e0e0',
            'grid': '#333333'
        }
        
        logger.info(f"Initialized interactive report generator with output dir: {self.report_output_dir}")
    
    def generate_report(self, 
                       backtest_result: Dict[str, Any],
                       performance_metrics: Dict[str, Any],
                       strategy_info: Dict[str, Any],
                       filename: Optional[str] = None) -> str:
        """
        Generate an interactive HTML report using Plotly.
        
        Args:
            backtest_result: The raw backtest result data
            performance_metrics: Calculated performance metrics
            strategy_info: Information about the strategy
            filename: Optional filename for the report
            
        Returns:
            str: Path to the generated report file
        """
        try:
            # Generate a report ID and filename if not provided
            report_id = create_unique_id()
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                strategy_name = strategy_info.get('name', 'unknown').replace(' ', '_')
                filename = f"{strategy_name}_{timestamp}_{report_id[:8]}_interactive"
            
            # Prepare data
            trades_df = pd.DataFrame(backtest_result.get('trades', []))
            equity_curve = backtest_result.get('equity_curve', pd.DataFrame())
            market_data = backtest_result.get('market_data', pd.DataFrame())
            
            # Generate visualizations as HTML
            visualizations = {}
            
            if not trades_df.empty and not equity_curve.empty:
                visualizations['equity_curve'] = self._generate_interactive_equity_curve(equity_curve)
                visualizations['drawdown_chart'] = self._generate_interactive_drawdown_chart(equity_curve)
                visualizations['trade_analysis'] = self._generate_interactive_trade_analysis(trades_df)
                
                if not market_data.empty:
                    visualizations['trades_on_price'] = self._generate_interactive_price_chart(
                        market_data, trades_df)
            
            # Prepare template data
            template_data = {
                'report_id': report_id,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'strategy_info': strategy_info,
                'performance_metrics': performance_metrics,
                'summary': self._prepare_summary_metrics(performance_metrics),
                'visualizations': visualizations,
                'trades': trades_df.to_dict('records') if not trades_df.empty else [],
                'backtest_params': backtest_result.get('params', {})
            }
            
            # Load the template
            template = self.jinja_env.get_template('interactive_report.html')
            
            # Render the template
            html_content = template.render(**template_data)
            
            # Save the HTML file
            file_path = os.path.join(self.report_output_dir, f"{filename}.html")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Generated interactive HTML report: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error generating interactive report: {e}")
            raise ReportGenerationError(f"Failed to generate interactive report: {e}")
    
    def _prepare_summary_metrics(self, 
                               performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare summary metrics for the report dashboard."""
        summary = {
            'total_return': performance_metrics.get('total_return', 0),
            'total_return_formatted': f"{performance_metrics.get('total_return', 0):.2f}%",
            'win_rate': performance_metrics.get('win_rate', 0),
            'win_rate_formatted': f"{performance_metrics.get('win_rate', 0):.2f}%",
            'profit_factor': performance_metrics.get('profit_factor', 0),
            'profit_factor_formatted': f"{performance_metrics.get('profit_factor', 0):.2f}",
            'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
            'sharpe_ratio_formatted': f"{performance_metrics.get('sharpe_ratio', 0):.2f}",
            'max_drawdown': performance_metrics.get('max_drawdown', 0),
            'max_drawdown_formatted': f"{performance_metrics.get('max_drawdown', 0):.2f}%",
            'total_trades': performance_metrics.get('total_trades', 0),
            'winning_trades': performance_metrics.get('winning_trades', 0),
            'losing_trades': performance_metrics.get('losing_trades', 0),
            'avg_trade': performance_metrics.get('avg_trade', 0),
            'avg_trade_formatted': f"{performance_metrics.get('avg_trade', 0):.2f}%",
            'avg_win': performance_metrics.get('avg_win', 0),
            'avg_win_formatted': f"{performance_metrics.get('avg_win', 0):.2f}%",
            'avg_loss': performance_metrics.get('avg_loss', 0),
            'avg_loss_formatted': f"{performance_metrics.get('avg_loss', 0):.2f}%",
        }
        
        return summary
    
    def _generate_interactive_equity_curve(self, equity_curve: pd.DataFrame) -> str:
        """Generate interactive equity curve visualization using Plotly."""
        try:
            if equity_curve.empty:
                return "No equity curve data available"

            # Create a subplot with 2 rows (equity and drawdown)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.05, 
                              row_heights=[0.7, 0.3])
            
            # Add equity curve trace
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index, 
                    y=equity_curve['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color=self.colors['equity'], width=2)
                ),
                row=1, col=1
            )
            
            # Add benchmark if available
            if 'benchmark' in equity_curve.columns:
                fig.add_trace(
                    go.Scatter(
                        x=equity_curve.index, 
                        y=equity_curve['benchmark'],
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='#9C27B0', width=2, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Add drawdown trace
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index, 
                    y=-equity_curve['drawdown_pct'] if 'drawdown_pct' in equity_curve.columns else None,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color=self.colors['drawdown'], width=1)
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Strategy Equity Curve and Drawdown',
                template='plotly_dark',
                height=700,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,1)',
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            # Update yaxis titles
            fig.update_yaxes(title_text="Equity", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            
            # Invert y-axis for drawdown (to show negative values as going down)
            fig.update_yaxes(autorange="reversed", row=2, col=1)
            
            # Convert to HTML
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error generating interactive equity curve: {e}")
            return f"Error generating equity curve: {e}"
    
    def _generate_interactive_drawdown_chart(self, equity_curve: pd.DataFrame) -> str:
        """Generate interactive drawdown chart using Plotly."""
        try:
            if equity_curve.empty or 'drawdown_pct' not in equity_curve.columns:
                return "No drawdown data available"

            # Create figure
            fig = go.Figure()
            
            # Add drawdown trace
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index, 
                    y=-equity_curve['drawdown_pct'],
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color=self.colors['drawdown'], width=2)
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Drawdown Analysis',
                template='plotly_dark',
                height=500,
                yaxis_title='Drawdown (%)',
                xaxis_title='Date',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,1)',
                yaxis=dict(autorange="reversed"),  # Invert y-axis
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            # Convert to HTML
            return fig.to_html(full_html=False, include_plotlyjs=False)
            
        except Exception as e:
            logger.error(f"Error generating interactive drawdown chart: {e}")
            return f"Error generating drawdown chart: {e}"
    
    def _generate_interactive_trade_analysis(self, trades_df: pd.DataFrame) -> str:
        """Generate interactive trade analysis charts using Plotly."""
        try:
            if trades_df.empty:
                return "No trades data available"

            # Create a subplot with 2x2 layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Trade Profit/Loss Distribution', 
                    'Profit by Trade Direction',
                    'Cumulative Performance', 
                    'Win/Loss Ratio'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # 1. Top Left: Trade Distribution Histogram
            winning_trades = trades_df[trades_df['profit_pct'] > 0]['profit_pct']
            losing_trades = trades_df[trades_df['profit_pct'] <= 0]['profit_pct']
            
            fig.add_trace(
                go.Histogram(
                    x=winning_trades,
                    name='Winning Trades',
                    marker_color=self.colors['profit'],
                    opacity=0.7,
                    bingroup='trades'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(
                    x=losing_trades,
                    name='Losing Trades',
                    marker_color=self.colors['loss'],
                    opacity=0.7,
                    bingroup='trades'
                ),
                row=1, col=1
            )
            
            # 2. Top Right: Profit by Trade Direction
            if 'direction' in trades_df.columns:
                long_trades = trades_df[trades_df['direction'] == 'long']
                short_trades = trades_df[trades_df['direction'] == 'short']
                
                direction_data = [
                    {
                        'direction': 'Long',
                        'win_rate': (len(long_trades[long_trades['profit_pct'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0,
                        'avg_profit': float(long_trades['profit_pct'].mean()) if not long_trades.empty else 0.0,
                    },
                    {
                        'direction': 'Short',
                        'win_rate': (len(short_trades[short_trades['profit_pct'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0,
                        'avg_profit': float(short_trades['profit_pct'].mean()) if not short_trades.empty else 0.0,
                    }
                ]
                dir_df = pd.DataFrame(direction_data)

                fig.add_trace(
                    go.Bar(
                        x=dir_df['direction'],
                        y=dir_df['avg_profit'],
                        name='Avg Profit (%)',
                        marker_color=[self.colors['profit'], self.colors['loss']]
                    ),
                    row=1, col=2
                )

                fig.add_trace(
                    go.Bar(
                        x=dir_df['direction'],
                        y=dir_df['win_rate'],
                        name='Win Rate (%)',
                        marker_color=self.colors['equity'],
                        opacity=0.6
                    ),
                    row=1, col=2
                )

            # 3. Bottom Left: Cumulative Performance
            trades_df = trades_df.sort_values('entry_time')
            trades_df['cum_profit'] = trades_df['profit_pct'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=trades_df['entry_time'],
                    y=trades_df['cum_profit'],
                    mode='lines',
                    name='Cumulative P/L',
                    line=dict(color=self.colors['equity'], width=2)
                ),
                row=2, col=1
            )

            # 4. Bottom Right: Win/Loss Ratio
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            fig.add_trace(
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[win_count, loss_count],
                    marker_colors=[self.colors['profit'], self.colors['loss']],
                    hole=0.4,
                    showlegend=False
                ),
                row=2, col=2
            )

            fig.update_layout(
                title='Trade Analysis',
                template='plotly_dark',
                height=800,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,1)',
                margin=dict(l=10, r=10, t=50, b=10)
            )

            return fig.to_html(full_html=False, include_plotlyjs=False)
        except Exception as e:
            logger.error(f"Error generating interactive trade analysis: {e}")
            return f"Error generating trade analysis: {e}"


# Backwards compatibility alias
Report = BacktestReport
