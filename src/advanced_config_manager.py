"""
Advanced Configuration Manager
===============================

Gestión completa de configuraciones de la aplicación:
- APIs y credenciales
- Estrategias y presets
- Integración con agentes IA
- Configuraciones de testing
- Seguridad y encriptación
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from cryptography.fernet import Fernet
import base64
from dotenv import load_dotenv


class AdvancedConfigManager:
    def __init__(self, config_file='config/app_config.json'):
        self.config_file = Path(config_file)
        self.config = {}
        self.logger = logging.getLogger(__name__)
        self.encryption_key = None

        # Load configuration
        self.load_config()

        # Setup encryption if enabled
        if self.config.get('security_settings', {}).get('encrypt_credentials', False):
            self._setup_encryption()

    def load_config(self) -> Dict:
        """Load main application configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                self.logger.info("Configuration loaded successfully")
            else:
                self.logger.warning("Config file not found, creating default")
                self._create_default_config()

            # Load environment variables for sensitive data
            load_dotenv()
            self._load_credentials_from_env()

            return self.config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False

    # ==================== API Configuration ====================

    def get_api_config(self, provider: str) -> Optional[Dict]:
        """Get API configuration for a provider"""
        apis = self.config.get('api_connections', {})
        return apis.get(provider)

    def set_api_credentials(self, provider: str, api_key: str, secret_key: str) -> bool:
        """Set API credentials for a provider"""
        try:
            if provider not in self.config.get('api_connections', {}):
                self.logger.error(f"Unknown API provider: {provider}")
                return False

            # Encrypt credentials if enabled
            if self.config['security_settings']['encrypt_credentials']:
                api_key = self._encrypt(api_key)
                secret_key = self._encrypt(secret_key)

            self.config['api_connections'][provider]['api_key'] = api_key
            self.config['api_connections'][provider]['secret_key'] = secret_key

            return self.save_config()
        except Exception as e:
            self.logger.error(f"Failed to set API credentials: {e}")
            return False

    def get_api_credentials(self, provider: str) -> Tuple[Optional[str], Optional[str]]:
        """Get decrypted API credentials"""
        try:
            api_config = self.get_api_config(provider)
            if not api_config:
                return None, None

            api_key = api_config.get('api_key', '')
            secret_key = api_config.get('secret_key', '')

            # Decrypt if encrypted
            if self.config['security_settings']['encrypt_credentials']:
                if api_key:
                    api_key = self._decrypt(api_key)
                if secret_key:
                    secret_key = self._decrypt(secret_key)

            return api_key, secret_key
        except Exception as e:
            self.logger.error(f"Failed to get API credentials: {e}")
            return None, None

    def get_default_api(self) -> Optional[str]:
        """Get the default API provider"""
        apis = self.config.get('api_connections', {})
        for provider, config in apis.items():
            if config.get('is_default', False) and config.get('enabled', False):
                return provider
        return None

    def set_default_api(self, provider: str) -> bool:
        """Set default API provider"""
        try:
            apis = self.config.get('api_connections', {})

            # Disable current default
            for p in apis:
                apis[p]['is_default'] = False

            # Set new default
            if provider in apis:
                apis[provider]['is_default'] = True
                apis[provider]['enabled'] = True
                return self.save_config()

            return False
        except Exception as e:
            self.logger.error(f"Failed to set default API: {e}")
            return False

    def list_available_apis(self) -> List[Dict]:
        """List all available API providers"""
        apis = self.config.get('api_connections', {})
        return [
            {
                'name': provider,
                'enabled': config.get('enabled', False),
                'is_default': config.get('is_default', False),
                'base_url': config.get('base_url', ''),
                'has_credentials': bool(config.get('api_key', ''))
            }
            for provider, config in apis.items()
        ]

    # ==================== Strategy Configuration ====================

    def get_strategy_settings(self) -> Dict:
        """Get strategy configuration settings"""
        return self.config.get('strategy_settings', {})

    def save_strategy_preset(self, strategy_name: str, preset_name: str, params: Dict) -> bool:
        """Save a strategy preset"""
        try:
            presets_dir = Path(self.config['strategy_settings']['presets_dir'])
            presets_dir.mkdir(parents=True, exist_ok=True)

            strategy_presets_dir = presets_dir / strategy_name
            strategy_presets_dir.mkdir(exist_ok=True)

            preset_data = {
                'strategy': strategy_name,
                'preset_name': preset_name,
                'params': params,
                'created': datetime.now().isoformat(),
                'modified': datetime.now().isoformat()
            }

            preset_file = strategy_presets_dir / f"{preset_name}.json"
            with open(preset_file, 'w') as f:
                json.dump(preset_data, f, indent=2)

            self.logger.info(f"Saved preset '{preset_name}' for strategy '{strategy_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save strategy preset: {e}")
            return False

    def load_strategy_preset(self, strategy_name: str, preset_name: str) -> Optional[Dict]:
        """Load a strategy preset"""
        try:
            presets_dir = Path(self.config['strategy_settings']['presets_dir'])
            preset_file = presets_dir / strategy_name / f"{preset_name}.json"

            if preset_file.exists():
                with open(preset_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load strategy preset: {e}")
            return None

    def list_strategy_presets(self, strategy_name: str) -> List[str]:
        """List all presets for a strategy"""
        try:
            presets_dir = Path(self.config['strategy_settings']['presets_dir']) / strategy_name
            if presets_dir.exists():
                return [f.stem for f in presets_dir.glob('*.json')]
            return []
        except Exception as e:
            self.logger.error(f"Failed to list strategy presets: {e}")
            return []

    def delete_strategy_preset(self, strategy_name: str, preset_name: str) -> bool:
        """Delete a strategy preset"""
        try:
            presets_dir = Path(self.config['strategy_settings']['presets_dir'])
            preset_file = presets_dir / strategy_name / f"{preset_name}.json"

            if preset_file.exists():
                preset_file.unlink()
                self.logger.info(f"Deleted preset '{preset_name}' for strategy '{strategy_name}'")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete strategy preset: {e}")
            return False

    # ==================== AI Integration ====================

    def get_ai_config(self) -> Dict:
        """Get AI integration configuration"""
        return self.config.get('ai_integration', {})

    def is_ai_enabled(self) -> bool:
        """Check if AI integration is enabled"""
        ai_config = self.get_ai_config()
        return ai_config.get('vscode_integration', {}).get('enabled', False)

    def get_active_agent(self) -> Optional[str]:
        """Get the currently active AI agent"""
        ai_config = self.get_ai_config()
        vscode_config = ai_config.get('vscode_integration', {})

        if vscode_config.get('enabled', False):
            return vscode_config.get('agent_type', 'copilot')
        return None

    def set_active_agent(self, agent_name: str) -> bool:
        """Set the active AI agent"""
        try:
            ai_config = self.config['ai_integration']

            if agent_name not in ai_config.get('available_agents', {}):
                self.logger.error(f"Unknown AI agent: {agent_name}")
                return False

            ai_config['vscode_integration']['agent_type'] = agent_name
            ai_config['vscode_integration']['enabled'] = True
            ai_config['available_agents'][agent_name]['enabled'] = True

            return self.save_config()
        except Exception as e:
            self.logger.error(f"Failed to set active agent: {e}")
            return False

    def list_available_agents(self) -> List[Dict]:
        """List all available AI agents"""
        ai_config = self.get_ai_config()
        agents = ai_config.get('available_agents', {})

        return [
            {
                'id': agent_id,
                'name': agent_config.get('name', ''),
                'enabled': agent_config.get('enabled', False),
                'capabilities': agent_config.get('capabilities', []),
                'has_credentials': bool(agent_config.get('api_key', ''))
            }
            for agent_id, agent_config in agents.items()
        ]

    def configure_agent(self, agent_id: str, api_key: str = None, endpoint: str = None) -> bool:
        """Configure an AI agent"""
        try:
            if agent_id not in self.config['ai_integration']['available_agents']:
                return False

            agent_config = self.config['ai_integration']['available_agents'][agent_id]

            if api_key:
                # Encrypt API key
                if self.config['security_settings']['encrypt_credentials']:
                    api_key = self._encrypt(api_key)
                agent_config['api_key'] = api_key

            if endpoint:
                agent_config['endpoint'] = endpoint

            return self.save_config()
        except Exception as e:
            self.logger.error(f"Failed to configure agent: {e}")
            return False

    def get_analysis_options(self) -> Dict:
        """Get AI analysis options"""
        return self.config.get('ai_integration', {}).get('analysis_options', {})

    def should_trigger_analysis(self, trigger_type: str) -> bool:
        """Check if analysis should be triggered for an event"""
        triggers = self.config.get('ai_integration', {}).get('analysis_triggers', {})
        return triggers.get(trigger_type, False)

    # ==================== Backtest Configuration ====================

    def get_backtest_settings(self) -> Dict:
        """Get backtest configuration"""
        return self.config.get('backtest_settings', {})

    def get_data_settings(self) -> Dict:
        """Get data configuration"""
        return self.config.get('data_settings', {})

    def get_analysis_settings(self) -> Dict:
        """Get analysis configuration"""
        return self.config.get('analysis_settings', {})

    # ==================== Testing Configuration ====================

    def get_testing_settings(self) -> Dict:
        """Get testing configuration"""
        return self.config.get('testing_settings', {})

    def should_run_tests(self) -> bool:
        """Check if tests should be run automatically"""
        return self.config.get('testing_settings', {}).get('auto_run_tests', False)

    # ==================== Security & Encryption ====================

    def _setup_encryption(self):
        """Setup encryption key"""
        try:
            key_file = Path(self.config['security_settings']['encryption_key_file'])

            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                # Secure the key file
                os.chmod(key_file, 0o600)

            self.cipher = Fernet(self.encryption_key)
        except Exception as e:
            self.logger.error(f"Failed to setup encryption: {e}")

    def _encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            if not self.encryption_key:
                return data
            encrypted = self.cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data

    def _decrypt(self, data: str) -> str:
        """Decrypt sensitive data"""
        try:
            if not self.encryption_key:
                return data
            decrypted = self.cipher.decrypt(base64.b64decode(data.encode()))
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return data

    def _load_credentials_from_env(self):
        """Load credentials from environment variables"""
        try:
            # Alpaca credentials
            alpaca_key = os.getenv('ALPACA_API_KEY')
            alpaca_secret = os.getenv('ALPACA_SECRET_KEY')

            if alpaca_key and alpaca_secret:
                self.config['api_connections']['alpaca']['api_key'] = alpaca_key
                self.config['api_connections']['alpaca']['secret_key'] = alpaca_secret
        except Exception as e:
            self.logger.error(f"Failed to load credentials from env: {e}")

    def _create_default_config(self):
        """Create default configuration - should load from app_config.json"""
        # This will be loaded from the JSON file
        pass

    # ==================== Utility Methods ====================

    def get_export_paths(self) -> Dict:
        """Get export paths configuration"""
        return self.config.get('export_paths', {})

    def get_notification_settings(self) -> Dict:
        """Get notification settings"""
        return self.config.get('notification_settings', {})

    def get_performance_settings(self) -> Dict:
        """Get performance optimization settings"""
        return self.config.get('performance_settings', {})

    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []

        # Check required sections
        required_sections = [
            'app', 'api_connections', 'data_settings',
            'backtest_settings', 'strategy_settings'
        ]

        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")

        # Check default API has credentials
        default_api = self.get_default_api()
        if default_api:
            api_key, secret_key = self.get_api_credentials(default_api)
            if not api_key or not secret_key:
                errors.append(f"Default API '{default_api}' missing credentials")

        return len(errors) == 0, errors

    def export_config_template(self, output_file: str) -> bool:
        """Export configuration template for documentation"""
        try:
            # Create a sanitized version without credentials
            template = self.config.copy()

            # Remove sensitive data
            for provider in template.get('api_connections', {}):
                template['api_connections'][provider]['api_key'] = ''
                template['api_connections'][provider]['secret_key'] = ''

            for agent in template.get('ai_integration', {}).get('available_agents', {}):
                if 'api_key' in template['ai_integration']['available_agents'][agent]:
                    template['ai_integration']['available_agents'][agent]['api_key'] = ''

            with open(output_file, 'w') as f:
                json.dump(template, f, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Failed to export config template: {e}")
            return False
