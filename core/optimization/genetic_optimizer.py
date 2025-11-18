"""
Genetic Algorithm Optimization Module
Implements genetic algorithms for strategy parameter optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

logger = logging.getLogger(__name__)

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population"""
    genes: Dict[str, Any]  # Parameter values
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

@dataclass
class OptimizationConfig:
    """Configuration for genetic algorithm optimization"""
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_count: int = 5
    tournament_size: int = 5
    max_workers: int = 4
    fitness_function: str = "sharpe_ratio"  # sharpe_ratio, total_return, calmar_ratio, sortino_ratio
    constraint_penalty: float = 1000.0

class GeneticOptimizer:
    """
    Genetic algorithm optimizer for strategy parameters.
    Uses evolutionary principles to find optimal parameter combinations.
    """

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self.parameter_types: Dict[str, str] = {}  # 'int', 'float', 'categorical'
        self.categorical_values: Dict[str, List[Any]] = {}
        self.fitness_history: List[float] = []
        self.best_individuals: List[Individual] = []

    def set_parameter_bounds(self, bounds: Dict[str, Tuple[float, float]],
                           param_types: Dict[str, str] = None,
                           categorical_values: Dict[str, List[Any]] = None):
        """
        Set parameter bounds and types for optimization

        Args:
            bounds: Dict mapping parameter names to (min, max) tuples
            param_types: Dict mapping parameter names to types ('int', 'float', 'categorical')
            categorical_values: Dict mapping categorical parameter names to possible values
        """
        self.parameter_bounds = bounds
        self.parameter_types = param_types or {}
        self.categorical_values = categorical_values or {}

        # Set default types to float if not specified
        for param in bounds.keys():
            if param not in self.parameter_types:
                self.parameter_types[param] = 'float'

    def initialize_population(self) -> List[Individual]:
        """Initialize random population"""
        population = []

        for _ in range(self.config.population_size):
            genes = {}
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                param_type = self.parameter_types.get(param_name, 'float')

                if param_type == 'int':
                    genes[param_name] = random.randint(int(min_val), int(max_val))
                elif param_type == 'categorical':
                    values = self.categorical_values.get(param_name, [min_val, max_val])
                    genes[param_name] = random.choice(values)
                else:  # float
                    genes[param_name] = random.uniform(min_val, max_val)

            population.append(Individual(genes=genes))

        return population

    def evaluate_fitness(self, individual: Individual, backtest_function: Callable) -> Individual:
        """
        Evaluate fitness of an individual by running backtest

        Args:
            individual: Individual to evaluate
            backtest_function: Function that takes parameters and returns backtest results

        Returns:
            Individual with updated fitness metrics
        """
        try:
            # Run backtest with individual's parameters
            results = backtest_function(**individual.genes)

            if not results or len(results) == 0:
                individual.fitness = -self.config.constraint_penalty
                return individual

            # Extract metrics from first (and typically only) result
            metrics = results[0].metrics

            # Store key metrics
            individual.sharpe_ratio = metrics.get('sharpe_ratio', 0)
            individual.total_return = metrics.get('total_return', 0)
            individual.max_drawdown = metrics.get('max_drawdown', 0)
            individual.win_rate = metrics.get('win_rate', 0)

            # Calculate fitness based on configured function
            if self.config.fitness_function == "sharpe_ratio":
                fitness = individual.sharpe_ratio
            elif self.config.fitness_function == "total_return":
                fitness = individual.total_return
            elif self.config.fitness_function == "calmar_ratio":
                fitness = metrics.get('calmar_ratio', 0)
            elif self.config.fitness_function == "sortino_ratio":
                fitness = metrics.get('sortino_ratio', 0)
            else:
                fitness = individual.sharpe_ratio  # default

            # Apply constraints (penalize high drawdown)
            if individual.max_drawdown > 0.3:  # 30% max drawdown constraint
                fitness -= self.config.constraint_penalty * (individual.max_drawdown - 0.3)

            individual.fitness = fitness

        except Exception as e:
            logger.error(f"Error evaluating individual {individual.genes}: {e}")
            individual.fitness = -self.config.constraint_penalty

        return individual

    def evaluate_population_parallel(self, population: List[Individual],
                                   backtest_function: Callable) -> List[Individual]:
        """Evaluate population fitness in parallel"""
        evaluated_population = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_individual = {
                executor.submit(self.evaluate_fitness, individual, backtest_function): individual
                for individual in population
            }

            # Collect results as they complete
            for future in as_completed(future_to_individual):
                try:
                    evaluated_individual = future.result()
                    evaluated_population.append(evaluated_individual)
                except Exception as e:
                    logger.error(f"Error in parallel evaluation: {e}")
                    # Add individual with penalty fitness
                    original_individual = future_to_individual[future]
                    original_individual.fitness = -self.config.constraint_penalty
                    evaluated_population.append(original_individual)

        return evaluated_population

    def select_parents_tournament(self, population: List[Individual]) -> List[Individual]:
        """Select parents using tournament selection"""
        parents = []

        for _ in range(len(population)):
            # Select tournament participants
            tournament = random.sample(population, min(self.config.tournament_size, len(population)))

            # Find winner (highest fitness)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)

        return parents

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents"""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2

        child1_genes = {}
        child2_genes = {}

        for param_name in self.parameter_bounds.keys():
            if random.random() < 0.5:
                # Swap values
                child1_genes[param_name] = parent1.genes[param_name]
                child2_genes[param_name] = parent2.genes[param_name]
            else:
                child1_genes[param_name] = parent2.genes[param_name]
                child2_genes[param_name] = parent1.genes[param_name]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual"""
        mutated_genes = individual.genes.copy()

        for param_name, value in mutated_genes.items():
            if random.random() < self.config.mutation_rate:
                bounds = self.parameter_bounds[param_name]
                param_type = self.parameter_types.get(param_name, 'float')

                if param_type == 'int':
                    # Random new value within bounds
                    mutated_genes[param_name] = random.randint(int(bounds[0]), int(bounds[1]))
                elif param_type == 'categorical':
                    values = self.categorical_values.get(param_name, [bounds[0], bounds[1]])
                    current_index = values.index(value) if value in values else 0
                    # Mutate to adjacent values
                    new_index = max(0, min(len(values) - 1,
                                          current_index + random.choice([-1, 1])))
                    mutated_genes[param_name] = values[new_index]
                else:  # float
                    # Add/subtract random amount (up to 20% of range)
                    range_size = bounds[1] - bounds[0]
                    mutation_amount = random.uniform(-0.2 * range_size, 0.2 * range_size)
                    new_value = value + mutation_amount
                    mutated_genes[param_name] = max(bounds[0], min(bounds[1], new_value))

        return Individual(genes=mutated_genes)

    def create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """Create next generation through selection, crossover, and mutation"""
        # Sort by fitness (descending)
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Elitism - keep best individuals
        next_generation = population[:self.config.elitism_count]

        # Select parents
        parents = self.select_parents_tournament(population)

        # Create offspring through crossover and mutation
        while len(next_generation) < self.config.population_size:
            # Select two parents
            parent1, parent2 = random.sample(parents, 2)

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            next_generation.extend([child1, child2])

        # Trim to population size
        return next_generation[:self.config.population_size]

    def optimize(self, backtest_function: Callable,
                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization

        Args:
            backtest_function: Function that takes parameters and returns backtest results
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with optimization results
        """
        logger.info("Starting genetic algorithm optimization")
        start_time = time.time()

        # Initialize population
        population = self.initialize_population()

        # Evaluate initial population
        if progress_callback:
            progress_callback("Evaluating initial population...")

        population = self.evaluate_population_parallel(population, backtest_function)

        best_fitness = max(ind.fitness for ind in population)
        self.fitness_history.append(best_fitness)
        self.best_individuals.append(max(population, key=lambda x: x.fitness))

        logger.info(f"Generation 0: Best fitness = {best_fitness:.4f}")

        # Evolutionary loop
        for generation in range(1, self.config.generations + 1):
            if progress_callback:
                progress_callback(f"Generation {generation}/{self.config.generations}...")

            # Create next generation
            population = self.create_next_generation(population)

            # Evaluate new population
            population = self.evaluate_population_parallel(population, backtest_function)

            # Track best individual
            best_individual = max(population, key=lambda x: x.fitness)
            self.fitness_history.append(best_individual.fitness)
            self.best_individuals.append(best_individual)

            logger.info(f"Generation {generation}: Best fitness = {best_individual.fitness:.4f}")

        # Find overall best
        best_individual = max(self.best_individuals, key=lambda x: x.fitness)

        optimization_time = time.time() - start_time

        results = {
            'best_parameters': best_individual.genes,
            'best_fitness': best_individual.fitness,
            'best_metrics': {
                'sharpe_ratio': best_individual.sharpe_ratio,
                'total_return': best_individual.total_return,
                'max_drawdown': best_individual.max_drawdown,
                'win_rate': best_individual.win_rate
            },
            'fitness_history': self.fitness_history,
            'optimization_time': optimization_time,
            'generations': self.config.generations,
            'population_size': self.config.population_size,
            'convergence_generation': np.argmax(self.fitness_history) + 1
        }

        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best fitness: {best_individual.fitness:.4f}")
        logger.info(f"Best parameters: {best_individual.genes}")

        return results

    def get_parameter_ranges_from_strategy(self, strategy_class) -> Dict[str, Any]:
        """
        Extract parameter ranges from strategy class
        Assumes strategy class has PARAM_RANGES class attribute
        """
        if hasattr(strategy_class, 'PARAM_RANGES'):
            return strategy_class.PARAM_RANGES
        else:
            # Default ranges for common parameters
            return {
                'fast_period': (5, 50),
                'slow_period': (10, 200),
                'signal_period': (5, 50),
                'stop_loss': (0.01, 0.20),
                'take_profit': (0.01, 0.50)
            }

class OptimizationManager:
    """
    High-level manager for parameter optimization across multiple strategies
    """

    def __init__(self):
        self.optimizers: Dict[str, GeneticOptimizer] = {}
        self.optimization_results: Dict[str, Dict] = {}

    def add_strategy_optimization(self, strategy_name: str,
                                strategy_class: Any,
                                config: Optional[OptimizationConfig] = None):
        """Add a strategy for optimization"""
        optimizer = GeneticOptimizer(config or OptimizationConfig())

        # Get parameter ranges from strategy
        param_ranges = optimizer.get_parameter_ranges_from_strategy(strategy_class)
        optimizer.set_parameter_bounds(param_ranges)

        self.optimizers[strategy_name] = optimizer

    def optimize_strategy(self, strategy_name: str,
                         backtest_function: Callable,
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Optimize a specific strategy"""
        if strategy_name not in self.optimizers:
            raise ValueError(f"Strategy {strategy_name} not found in optimizers")

        optimizer = self.optimizers[strategy_name]
        results = optimizer.optimize(backtest_function, progress_callback)

        self.optimization_results[strategy_name] = results
        return results

    def optimize_all_strategies(self, backtest_functions: Dict[str, Callable],
                              progress_callback: Optional[Callable] = None) -> Dict[str, Dict]:
        """Optimize all registered strategies"""
        results = {}

        for strategy_name, backtest_func in backtest_functions.items():
            if progress_callback:
                progress_callback(f"Optimizing {strategy_name}...")

            try:
                strategy_results = self.optimize_strategy(strategy_name, backtest_func)
                results[strategy_name] = strategy_results
            except Exception as e:
                logger.error(f"Error optimizing {strategy_name}: {e}")
                results[strategy_name] = {'error': str(e)}

        return results

    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of all optimization results"""
        summary_data = []

        for strategy_name, results in self.optimization_results.items():
            if 'error' in results:
                continue

            row = {
                'Strategy': strategy_name,
                'Best Fitness': results['best_fitness'],
                'Sharpe Ratio': results['best_metrics']['sharpe_ratio'],
                'Total Return': results['best_metrics']['total_return'],
                'Max Drawdown': results['best_metrics']['max_drawdown'],
                'Win Rate': results['best_metrics']['win_rate'],
                'Optimization Time': results['optimization_time'],
                'Convergence Gen': results['convergence_generation']
            }
            summary_data.append(row)

        return pd.DataFrame(summary_data)