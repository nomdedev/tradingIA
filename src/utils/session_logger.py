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
    
    def log_error(self, error_type, error_message, context=None):
        """Log an error with details"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'context': context or {},
            'stack_trace': None  # Could add traceback if needed
        }
        
        self.session_data['errors'].append(error_entry)
        self._auto_save()
    
    def log_backtest(self, strategy, ticker, timeframe, results):
        """Log a backtest execution"""
        self.session_data['results']['backtests_run'] += 1
        
        if strategy not in self.session_data['results']['strategies_tested']:
            self.session_data['results']['strategies_tested'].append(strategy)
        
        self.log_action('backtest', {
            'strategy': strategy,
            'ticker': ticker,
            'timeframe': timeframe,
            'results': results
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
                
                # Errors
                if self.session_data['errors']:
                    f.write("-" * 80 + "\n")
                    f.write("ERRORS ENCOUNTERED\n")
                    f.write("-" * 80 + "\n")
                    for i, error in enumerate(self.session_data['errors'], 1):
                        f.write(f"\nError #{i}:\n")
                        f.write(f"  Time: {error['timestamp']}\n")
                        f.write(f"  Type: {error['type']}\n")
                        f.write(f"  Message: {error['message']}\n")
                        if error.get('context'):
                            f.write(f"  Context: {error['context']}\n")
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
                            f.write(f"  {key}: {value}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
                
        except Exception as e:
            print(f"Error creating text report: {e}")
