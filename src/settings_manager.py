import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class SettingsManager:
    def __init__(self, config_dir='config'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_config(self, config_dict: Dict[str, Any]) -> bool:
        """Save main configuration"""
        try:
            config_file = self.config_dir / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            self.logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False

    def load_config(self) -> Dict[str, Any]:
        """Load main configuration"""
        try:
            config_file = self.config_dir / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def save_preset(self,
                    strategy_name: str,
                    params_dict: Dict[str,
                                      Any],
                    preset_name: str) -> bool:
        """Save strategy preset"""
        try:
            presets_dir = self.config_dir / 'presets' / strategy_name
            presets_dir.mkdir(parents=True, exist_ok=True)

            preset_data = {
                'strategy': strategy_name,
                'params': params_dict,
                'created': datetime.now().isoformat(),
                'name': preset_name
            }

            preset_file = presets_dir / f"{preset_name}.json"
            with open(preset_file, 'w') as f:
                json.dump(preset_data, f, indent=2, default=str)

            self.logger.info(f"Preset {preset_name} saved for {strategy_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save preset: {e}")
            return False

    def load_preset(self, strategy_name: str, preset_name: str) -> Optional[Dict[str, Any]]:
        """Load strategy preset"""
        try:
            preset_file = self.config_dir / 'presets' / strategy_name / f"{preset_name}.json"
            if preset_file.exists():
                with open(preset_file, 'r') as f:
                    return json.load(f)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to load preset: {e}")
            return None

    def list_presets(self, strategy_name: str) -> List[str]:
        """List available presets for a strategy"""
        try:
            presets_dir = self.config_dir / 'presets' / strategy_name
            if presets_dir.exists():
                return [f.stem for f in presets_dir.glob('*.json')]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Failed to list presets: {e}")
            return []

    def save_backtest_result(
            self,
            strategy: str,
            params: Dict,
            metrics: Dict,
            trades: List[Dict]) -> bool:
        """Save backtest result to database"""
        try:
            results_dir = self.config_dir / 'backtest_results'
            results_dir.mkdir(exist_ok=True)

            result_data = {
                'strategy': strategy,
                'params': params,
                'metrics': metrics,
                'trades': trades,
                'timestamp': datetime.now().isoformat(),
                'id': f"{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }

            result_file = results_dir / f"{result_data['id']}.json"
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

            self.logger.info(f"Backtest result saved: {result_data['id']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save backtest result: {e}")
            return False

    def get_recent_results(self, limit: int = 5) -> List[Dict]:
        """Get recent backtest results"""
        try:
            results_dir = self.config_dir / 'backtest_results'
            if not results_dir.exists():
                return []

            result_files = list(results_dir.glob('*.json'))
            result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            results = []
            for file_path in result_files[:limit]:
                try:
                    with open(file_path, 'r') as f:
                        result = json.load(f)
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to load result {file_path}: {e}")

            return results
        except Exception as e:
            self.logger.error(f"Failed to get recent results: {e}")
            return []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'api_key': '',
            'secret_key': '',
            'default_symbol': 'BTC-USD',
            'default_timeframe': '5Min',
            'theme': 'dark',
            'auto_save': True,
            'max_threads': 4
        }
