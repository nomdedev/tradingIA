"""
Session Logger
Logs all user actions and results during a session for later analysis
"""

import json
import os
from datetime import datetime


class SessionLogger:
    """Logs user session activities and results"""
    
    def __init__(self, reports_dir='reports/sessions'):
        self.reports_dir = reports_dir
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_file = os.path.join(reports_dir, f'session_{self.session_id}.json')
        
        # Ensure reports directory exists
        os.makedirs(reports_dir, exist_ok=True)
        
        # Session data structure
        self.session_data = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'user_actions': [],
            'errors': [],
            'results': {
                'backtests_run': 0,
                'live_trading_sessions': 0,
                'data_downloads': 0,
                'strategies_tested': []
            },
            'configuration_changes': [],
            'tab_visits': {}
        }
    
    def log_action(self, action_type, details, result='success'):
        """Log a user action"""
        action_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': action_type,
            'details': details,
            'result': result
        }
        
        self.session_data['user_actions'].append(action_entry)
        self._auto_save()
    
    def log_error(self, error_type, error_message, context=None, stack_trace=None):
        """Log an error with detailed technical information"""
        import platform
        import sys
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'context': context or {},
            'stack_trace': stack_trace,
            'system_info': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'architecture': platform.architecture()
            },
            'session_info': {
                'session_id': self.session_id,
                'actions_count': len(self.session_data['user_actions']),
                'errors_count': len(self.session_data['errors'])
            }
        }
        
        self.session_data['errors'].append(error_entry)
        self._auto_save()
    
    def log_backtest(self, strategy, ticker, timeframe, results):
        """Log a backtest execution with detailed metrics"""
        self.session_data['results']['backtests_run'] += 1
        
        if strategy not in self.session_data['results']['strategies_tested']:
            self.session_data['results']['strategies_tested'].append(strategy)
        
        # Extract key metrics for summary
        key_metrics = {}
        if isinstance(results, dict) and 'metrics' in results:
            metrics = results['metrics']
            key_metrics = {
                'sharpe_ratio': metrics.get('sharpe', 0),
                'total_return': metrics.get('total_return', 0),
                'win_rate': metrics.get('win_rate', 0),
                'max_drawdown': metrics.get('max_dd', 0),
                'total_trades': metrics.get('num_trades', 0),
                'profit_factor': metrics.get('profit_factor', 0)
            }
        
        self.log_action('backtest', {
            'strategy': strategy,
            'ticker': ticker,
            'timeframe': timeframe,
            'key_metrics': key_metrics,
            'has_strategy_params': 'strategy_parameters' in results if isinstance(results, dict) else False,
            'results_summary': {
                'trades_count': len(results.get('trades', [])) if isinstance(results, dict) else 0,
                'equity_points': len(results.get('equity_curve', [])) if isinstance(results, dict) else 0,
                'error': 'error' in results if isinstance(results, dict) else False
            }
        })
    
    def log_live_trading_session(self, ticker, strategy, duration_seconds, pnl, trades_count):
        """Log a live trading session"""
        self.session_data['results']['live_trading_sessions'] += 1
        
        self.log_action('live_trading', {
            'ticker': ticker,
            'strategy': strategy,
            'duration_seconds': duration_seconds,
            'final_pnl': pnl,
            'trades_executed': trades_count
        })
    
    def log_data_download(self, timeframe, status, file_path=None, error=None):
        """Log a data download attempt"""
        self.session_data['results']['data_downloads'] += 1
        
        self.log_action('data_download', {
            'timeframe': timeframe,
            'status': status,
            'file_path': file_path,
            'error': error
        }, result=status)
    
    def log_config_change(self, config_key, old_value, new_value):
        """Log a configuration change"""
        change_entry = {
            'timestamp': datetime.now().isoformat(),
            'config_key': config_key,
            'old_value': str(old_value),
            'new_value': str(new_value)
        }
        
        self.session_data['configuration_changes'].append(change_entry)
        self._auto_save()
    
    def log_tab_visit(self, tab_name):
        """Log a tab visit"""
        if tab_name not in self.session_data['tab_visits']:
            self.session_data['tab_visits'][tab_name] = 0
        
        self.session_data['tab_visits'][tab_name] += 1
        self._auto_save()
    
    def log_ui_event(self, event_type, details):
        """Log any UI event (button clicks, layout changes, etc)"""
        self.log_action(f'ui_{event_type}', details)
    
    def log_window_event(self, event_type, window_info):
        """Log window geometry or state changes"""
        self.log_action('window_event', {
            'event_type': event_type,
            'window_info': window_info
        })
    
    def log_component_load_failure(self, component_name, error_message, stack_trace=None):
        """Log when a platform component fails to load"""
        self.log_error(
            error_type='component_load_failure',
            error_message=f"{component_name} failed to load: {error_message}",
            context={'component': component_name},
            stack_trace=stack_trace
        )
    
    def log_data_loading_event(self, event_type, data_info):
        """Log data loading events (start, progress, completion, failure)"""
        self.log_action('data_loading', {
            'event_type': event_type,
            'data_info': data_info
        })
    
    def end_session(self):
        """Finalize and save session report"""
        self.session_data['end_time'] = datetime.now().isoformat()
        
        # Calculate session duration
        start = datetime.fromisoformat(self.session_data['start_time'])
        end = datetime.fromisoformat(self.session_data['end_time'])
        duration = (end - start).total_seconds()
        
        self.session_data['duration_seconds'] = duration
        self.session_data['duration_human'] = self._format_duration(duration)
        
        # Generate summary
        self.session_data['summary'] = self._generate_summary()
        
        # Save final report
        self._save_session()
        
        # Also create a human-readable text version
        self._create_text_report()
    
    def _generate_summary(self):
        """Generate session summary"""
        return {
            'total_actions': len(self.session_data['user_actions']),
            'total_errors': len(self.session_data['errors']),
            'most_visited_tab': self._get_most_visited_tab(),
            'most_tested_strategy': self._get_most_tested_strategy(),
            'error_rate': len(self.session_data['errors']) / max(1, len(self.session_data['user_actions'])),
            'backtests_completed': self.session_data['results']['backtests_run'],
            'live_sessions': self.session_data['results']['live_trading_sessions'],
            'data_downloads': self.session_data['results']['data_downloads']
        }
    
    def _get_most_visited_tab(self):
        """Get the most visited tab"""
        if not self.session_data['tab_visits']:
            return 'None'
        
        return max(self.session_data['tab_visits'].items(), key=lambda x: x[1])[0]
    
    def _get_most_tested_strategy(self):
        """Get the most tested strategy"""
        strategies = self.session_data['results']['strategies_tested']
        return strategies[0] if strategies else 'None'
    
    def _format_duration(self, seconds):
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def _auto_save(self):
        """Auto-save session data (lightweight saves)"""
        # Only save every 10 actions to avoid excessive I/O
        if len(self.session_data['user_actions']) % 10 == 0:
            self._save_session()
    
    def _save_session(self):
        """Save session data to JSON file"""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            print(f"Error saving session: {e}")
    
    def _create_text_report(self):
        """Create human-readable text report"""
        text_file = self.session_file.replace('.json', '.txt')
        
        try:
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("TRADING PLATFORM - SESSION REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Start Time: {self.session_data['start_time']}\n")
                f.write(f"End Time: {self.session_data['end_time']}\n")
                f.write(f"Duration: {self.session_data.get('duration_human', 'N/A')}\n\n")
                
                # Summary
                f.write("-" * 80 + "\n")
                f.write("SUMMARY\n")
                f.write("-" * 80 + "\n")
                summary = self.session_data.get('summary', {})
                f.write(f"Total Actions: {summary.get('total_actions', 0)}\n")
                f.write(f"Total Errors: {summary.get('total_errors', 0)}\n")
                f.write(f"Error Rate: {summary.get('error_rate', 0):.2%}\n")
                f.write(f"Backtests Run: {summary.get('backtests_completed', 0)}\n")
                f.write(f"Live Trading Sessions: {summary.get('live_sessions', 0)}\n")
                f.write(f"Data Downloads: {summary.get('data_downloads', 0)}\n")
                f.write(f"Most Visited Tab: {summary.get('most_visited_tab', 'N/A')}\n\n")
                
                # Tab Visits
                f.write("-" * 80 + "\n")
                f.write("TAB VISITS\n")
                f.write("-" * 80 + "\n")
                for tab, count in sorted(self.session_data['tab_visits'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {tab}: {count} visits\n")
                f.write("\n")
                
                # Errors with technical details
                if self.session_data['errors']:
                    f.write("-" * 80 + "\n")
                    f.write("TECHNICAL ERRORS & DIAGNOSTICS\n")
                    f.write("-" * 80 + "\n")
                    for i, error in enumerate(self.session_data['errors'], 1):
                        f.write(f"\n🚨 ERROR #{i} - {error['type'].upper()}\n")
                        f.write(f"   Time: {error['timestamp']}\n")
                        f.write(f"   Message: {error['message']}\n")
                        
                        if error.get('context'):
                            f.write(f"   Context:\n")
                            for key, value in error['context'].items():
                                f.write(f"     {key}: {value}\n")
                        
                        if error.get('system_info'):
                            sys_info = error['system_info']
                            f.write(f"   System: {sys_info.get('platform', 'Unknown')}\n")
                            f.write(f"   Python: {sys_info.get('python_version', 'Unknown').split()[0]}\n")
                        
                        if error.get('stack_trace'):
                            f.write(f"   Stack Trace: {error['stack_trace'][:200]}...\n")
                        
                        f.write("\n")
                
                # Backtest Performance Summary
                backtests = [action for action in self.session_data['user_actions'] 
                           if action['type'] == 'backtest']
                if backtests:
                    f.write("-" * 80 + "\n")
                    f.write("BACKTEST PERFORMANCE SUMMARY\n")
                    f.write("-" * 80 + "\n")
                    
                    total_backtests = len(backtests)
                    successful_backtests = sum(1 for bt in backtests if bt.get('result') == 'success')
                    avg_sharpe = sum(bt.get('details', {}).get('key_metrics', {}).get('sharpe_ratio', 0) 
                                   for bt in backtests) / max(1, total_backtests)
                    
                    f.write(f"Total Backtests: {total_backtests}\n")
                    f.write(f"Successful: {successful_backtests} ({successful_backtests/total_backtests*100:.1f}%)\n")
                    f.write(f"Average Sharpe Ratio: {avg_sharpe:.3f}\n")
                    
                    # Best performing strategy
                    best_bt = max(backtests, 
                                key=lambda x: x.get('details', {}).get('key_metrics', {}).get('sharpe_ratio', -999))
                    best_metrics = best_bt.get('details', {}).get('key_metrics', {})
                    f.write(f"\n🏆 Best Backtest:\n")
                    f.write(f"   Strategy: {best_bt.get('details', {}).get('strategy', 'Unknown')}\n")
                    f.write(f"   Sharpe: {best_metrics.get('sharpe_ratio', 0):.3f}\n")
                    f.write(f"   Win Rate: {best_metrics.get('win_rate', 0):.1%}\n")
                    f.write(f"   Total Return: {best_metrics.get('total_return', 0):.2%}\n")
                    f.write("\n")
                
                # Data Operations Summary
                data_actions = [action for action in self.session_data['user_actions'] 
                              if 'data' in action['type'].lower()]
                if data_actions:
                    f.write("-" * 80 + "\n")
                    f.write("DATA OPERATIONS SUMMARY\n")
                    f.write("-" * 80 + "\n")
                    
                    downloads = sum(1 for action in data_actions if action['type'] == 'data_download')
                    successful_downloads = sum(1 for action in data_actions 
                                             if action['type'] == 'data_download' and action.get('result') == 'success')
                    
                    f.write(f"Data Downloads Attempted: {downloads}\n")
                    f.write(f"Successful Downloads: {successful_downloads}")
                    if downloads > 0:
                        f.write(f" ({successful_downloads/downloads*100:.1f}%)")
                    f.write("\n")
                    
                    # Check for data-related errors
                    data_errors = [error for error in self.session_data['errors'] 
                                 if 'data' in error['type'].lower() or 'alpaca' in error['message'].lower()]
                    if data_errors:
                        f.write(f"Data-Related Errors: {len(data_errors)}\n")
                        for error in data_errors[-3:]:  # Show last 3
                            f.write(f"  • {error['type']}: {error['message'][:100]}...\n")
                    f.write("\n")
                
                # Recent Actions (last 20)
                f.write("-" * 80 + "\n")
                f.write("RECENT ACTIONS (Last 20)\n")
                f.write("-" * 80 + "\n")
                recent_actions = self.session_data['user_actions'][-20:]
                for action in recent_actions:
                    f.write(f"\n[{action['timestamp']}] {action['type'].upper()}\n")
                    f.write(f"  Result: {action['result']}\n")
                    if action.get('details'):
                        for key, value in action['details'].items():
                            if key == 'key_metrics' and isinstance(value, dict):
                                f.write(f"  Key Metrics:\n")
                                for metric_key, metric_value in value.items():
                                    f.write(f"    {metric_key}: {metric_value}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
                
        except Exception as e:
            print(f"Error creating text report: {e}")
