"""
Strategy Registry for Dynamic Strategy Loading and Management
"""

from typing import Dict, Type, Any, List, Optional
import importlib
import pkgutil
import inspect
import logging
from .base_strategy import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)

class StrategyRegistry:
    """
    Registry for managing and loading trading strategies dynamically.

    This class provides functionality to:
    - Register strategy classes
    - Load strategies from modules
    - Create strategy instances from configurations
    - Discover available strategies
    """

    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._strategy_configs: Dict[str, Dict[str, Any]] = {}

    def register_strategy(self, strategy_class: Type[BaseStrategy],
                         config_template: Optional[Dict[str, Any]] = None):
        """
        Register a strategy class in the registry.

        Args:
            strategy_class: The strategy class to register
            config_template: Optional configuration template
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"Strategy class {strategy_class} must inherit from BaseStrategy")

        strategy_name = strategy_class.__name__
        self._strategies[strategy_name] = strategy_class

        if config_template:
            self._strategy_configs[strategy_name] = config_template

        logger.info(f"Registered strategy: {strategy_name}")

    def get_strategy_class(self, name: str) -> Type[BaseStrategy]:
        """
        Get a strategy class by name.

        Args:
            name: Name of the strategy class

        Returns:
            The strategy class

        Raises:
            ValueError: If strategy is not registered
        """
        if name not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available: {available}")

        return self._strategies[name]

    def create_strategy(self, config: StrategyConfig) -> BaseStrategy:
        """
        Create a strategy instance from configuration.

        Args:
            config: Strategy configuration

        Returns:
            Instantiated strategy object
        """
        strategy_class = self.get_strategy_class(config.name)
        return strategy_class(config)

    def list_strategies(self) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a registered strategy.

        Args:
            name: Strategy name

        Returns:
            Dictionary with strategy information
        """
        strategy_class = self.get_strategy_class(name)

        # Get required parameters by inspecting the class method
        try:
            # Create a dummy config with empty parameters to get required params
            dummy_config = StrategyConfig(
                name=name,
                description="Dummy config for introspection",
                parameters={}
            )
            # Create instance without validation to get required parameters
            dummy_strategy = object.__new__(strategy_class)
            dummy_strategy.config = dummy_config
            required_params = dummy_strategy.get_required_parameters()
        except Exception:
            required_params = []

        return {
            'name': name,
            'class': strategy_class.__name__,
            'module': strategy_class.__module__,
            'required_parameters': required_params,
            'config_template': self._strategy_configs.get(name, {}),
            'docstring': strategy_class.__doc__ or ""
        }

    def load_strategies_from_module(self, module_name: str):
        """
        Load all strategy classes from a Python module.

        Args:
            module_name: Name of the module to load from
        """
        try:
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, BaseStrategy) and
                    obj != BaseStrategy):
                    self.register_strategy(obj)
                    logger.info(f"Loaded strategy {name} from {module_name}")

        except ImportError as e:
            logger.error(f"Failed to load module {module_name}: {e}")

    def discover_strategies(self, package_name: str = "core.strategies"):
        """
        Discover and load all strategies from a package.

        Args:
            package_name: Name of the package to search in
        """
        try:
            package = importlib.import_module(package_name)

            if hasattr(package, '__path__'):
                # It's a package, discover submodules
                for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                    full_modname = f"{package_name}.{modname}"
                    if not ispkg and modname != "__init__":
                        self.load_strategies_from_module(full_modname)
            else:
                # It's a module, load directly
                self.load_strategies_from_module(package_name)

        except ImportError as e:
            logger.error(f"Failed to discover strategies in {package_name}: {e}")

    def create_strategy_from_dict(self, config_dict: Dict[str, Any]) -> BaseStrategy:
        """
        Create a strategy from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary with keys:
                - name: Strategy class name
                - description: Strategy description
                - parameters: Strategy parameters
                - risk_management: Risk management settings
                - filters: List of filters to apply

        Returns:
            Instantiated strategy object
        """
        config = StrategyConfig(
            name=config_dict['name'],
            description=config_dict.get('description', ''),
            parameters=config_dict.get('parameters', {}),
            risk_management=config_dict.get('risk_management', {}),
            filters=config_dict.get('filters', [])
        )

        return self.create_strategy(config)

    def get_default_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Default configuration dictionary
        """
        strategy_class = self.get_strategy_class(strategy_name)
        info = self.get_strategy_info(strategy_name)

        return {
            'name': strategy_name,
            'description': f"Default configuration for {strategy_name}",
            'parameters': {param: None for param in info['required_parameters']},
            'risk_management': {},
            'filters': []
        }

# Global registry instance
strategy_registry = StrategyRegistry()

def get_registry() -> StrategyRegistry:
    """Get the global strategy registry instance."""
    return strategy_registry

def register_strategy(strategy_class: Type[BaseStrategy],
                     config_template: Optional[Dict[str, Any]] = None):
    """Convenience function to register a strategy globally."""
    strategy_registry.register_strategy(strategy_class, config_template)

def create_strategy(config: StrategyConfig) -> BaseStrategy:
    """Convenience function to create a strategy globally."""
    return strategy_registry.create_strategy(config)