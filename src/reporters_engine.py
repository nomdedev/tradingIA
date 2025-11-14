import pandas as pd
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class ReportersEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def export_trades_csv(self, trades_df: pd.DataFrame, filename: str) -> bool:
        """Export trades to CSV file"""
        try:
            trades_df.to_csv(filename, index=False)
            self.logger.info(f"Trades exported to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export trades: {e}")
            return False

    def export_metrics_json(self, metrics_dict: Dict[str, Any], filename: str) -> bool:
        """Export metrics to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
            self.logger.info(f"Metrics exported to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False

    def export_to_pine_script(self, strategy_name: str,
                              params_dict: Dict[str, Any], output_file: str) -> bool:
        """Generate Pine Script v5 code with embedded parameters"""
        try:
            # Template for Pine Script
            pine_template = f'''//@version=5
indicator("{strategy_name} - Trading Strategy", overlay=true)

// Strategy Parameters
'''
            # Add parameters
            for param_name, param_value in params_dict.items():
                if isinstance(param_value, bool):
                    pine_template += f'{param_name} = {str(param_value).lower()}\n'
                elif isinstance(param_value, str):
                    pine_template += f'{param_name} = "{param_value}"\n'
                else:
                    pine_template += f'{param_name} = {param_value}\n'

            pine_template += '''
// Strategy Logic (placeholder - implement actual logic)
plot(close, color=color.blue, title="Close")
'''

            with open(output_file, 'w') as f:
                f.write(pine_template)

            self.logger.info(f"Pine Script exported to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export Pine Script: {e}")
            return False

    def generate_pdf_report(self, title: str, metrics: Dict[str, Any], trades: List[Dict],
                            charts_dict: Dict[str, Any], filename: str) -> bool:
        """Generate comprehensive PDF report"""
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
            )
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))

            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            summary_text = f"""
            Strategy Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
            Total Trades: {metrics.get('num_trades', 0)}<br/>
            Win Rate: {metrics.get('win_rate', 0):.1%}<br/>
            Sharpe Ratio: {metrics.get('sharpe', 0):.3f}<br/>
            Max Drawdown: {metrics.get('max_dd', 0):.1%}<br/>
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 12))

            # Metrics Table
            story.append(Paragraph("Performance Metrics", styles['Heading2']))

            metrics_data = [
                ['Metric', 'Value'],
                ['Sharpe Ratio', '.3f'],
                ['Calmar Ratio', '.3f'],
                ['Win Rate', '.1%'],
                ['Max Drawdown', '.1%'],
                ['Profit Factor', '.2f'],
                ['Total Return', '.1%'],
                ['Number of Trades', str(metrics.get('num_trades', 0))]
            ]

            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 12))

            # Recent Trades
            if trades:
                story.append(Paragraph("Recent Trades", styles['Heading2']))

                # Take last 10 trades
                recent_trades = trades[-10:] if len(trades) > 10 else trades

                trades_data = [['Timestamp', 'Entry', 'Exit', 'PnL%', 'Score']]
                for trade in recent_trades:
                    trades_data.append([
                        str(trade.get('timestamp', ''))[:19],
                        '.2f',
                        '.2f',
                        '.2f',
                        str(trade.get('score', 0))
                    ])

                trades_table = Table(trades_data)
                trades_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(trades_table)

            # Build PDF
            doc.build(story)
            self.logger.info(f"PDF report generated: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            return False

    def export_equity_curve_json(self, equity_list: List[float], filename: str) -> bool:
        """Export equity curve to JSON"""
        try:
            data = {
                'equity_curve': equity_list,
                'timestamps': [datetime.now().isoformat()] * len(equity_list),  # Placeholder
                'exported_at': datetime.now().isoformat()
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.info(f"Equity curve exported to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export equity curve: {e}")
            return False

    def generate_chart_image(self, fig, filename: str, format: str = 'png') -> bool:
        """Generate chart image from plotly figure"""
        try:
            if format == 'png':
                fig.write_image(filename)
            elif format == 'html':
                fig.write_html(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Chart saved as {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate chart image: {e}")
            return False
