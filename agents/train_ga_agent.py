import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import random
import os
import sys
from datetime import datetime
import joblib

# Añadir el directorio padre al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.indicators import add_all_indicators
from utils.advanced_indicators import add_all_advanced_indicators

class GAStrategy:
    """
    Estrategia GA Avanzada con 15 genes para trading institucional
    Incluye indicadores profesionales: Squeeze Momentum, IFVG, EMAs institucionales
    """

    def __init__(self, genes: List[float]):
        """
        Inicializa estrategia con 15 genes:
        0-1: RSI buy/sell thresholds
        2: MACD threshold
        3: Squeeze momentum threshold
        4: Use IFVG filter (0/1)
        5: EMA alignment required (0/1)
        6: Stop loss %
        7: Take profit %
        8-14: Pesos para diferentes señales (6 pesos)
        """
        if len(genes) != 15:
            raise ValueError("La estrategia GA requiere exactamente 15 genes")

        self.rsi_buy = max(10, min(50, genes[0]))  # RSI compra: 10-50
        self.rsi_sell = max(50, min(90, genes[1]))  # RSI venta: 50-90
        self.macd_threshold = genes[2]  # Umbral MACD
        self.squeeze_threshold = genes[3]  # Umbral squeeze momentum
        self.use_ifvg = genes[4] > 0.5  # Usar filtro IFVG
        self.ema_alignment_required = genes[5] > 0.5  # Requerir alineación EMAs
        self.stop_loss = max(0.5, min(5.0, genes[6]))  # Stop loss: 0.5-5%
        self.take_profit = max(1.0, min(10.0, genes[7]))  # Take profit: 1-10%

        # Pesos para diferentes señales (7 pesos)
        weights = genes[8:15]
        total_weight = sum(weights)
        if total_weight == 0:
            self.weights = [1/7] * 7  # Pesos iguales si todos son 0
        else:
            self.weights = [w/total_weight for w in weights]

        # Mapeo de pesos a señales
        self.weight_macd = self.weights[0]
        self.weight_volume = self.weights[1]
        self.weight_squeeze = self.weights[2]
        self.weight_ifvg = self.weights[3]
        self.weight_ema = self.weights[4]
        self.weight_momentum = self.weights[5]
        self.weight_trend = self.weights[6]

    def decide(self, row: pd.Series, position: int) -> Tuple[str, str]:
        """
        Toma decisión de trading basada en indicadores avanzados

        Args:
            row: Fila de datos con indicadores
            position: Posición actual (0 = sin posición, >0 = long)

        Returns:
            Tuple[str, str]: (decisión, razón)
        """
        # Filtros obligatorios institucionales
        if not self._passes_mandatory_filters(row):
            if position > 0:
                return "SELL", "VENTA - Filtros institucionales fallidos"
            return "HOLD", "HOLD - Filtros institucionales no cumplidos"

        # Si tenemos posición abierta, verificar stop loss/take profit
        if position > 0:
            entry_price = row.get('entry_price', row['Close'])
            current_price = row['Close']
            pnl_pct = (current_price - entry_price) / entry_price * 100

            if pnl_pct <= -self.stop_loss:
                return "SELL", ".1f"
            elif pnl_pct >= self.take_profit:
                return "SELL", ".1f"

        # Calcular score de compra/venta
        buy_score, sell_score = self._calculate_signal_scores(row)

        # Lógica de decisión
        if position == 0:  # Sin posición
            if buy_score >= 4:  # Al menos 4/7 señales positivas
                return "BUY", f"COMPRA (Score: {buy_score}/7) - {self._get_buy_reasons(row)}"
            else:
                return "HOLD", f"HOLD - Score insuficiente: {buy_score}/7"

        else:  # Con posición
            if sell_score >= 3:  # Al menos 3/7 señales de venta
                return "SELL", f"VENTA (Score: {sell_score}/7) - {self._get_sell_reasons(row)}"
            else:
                return "HOLD", "HOLD - Posición abierta, sin señales de venta"

    def _passes_mandatory_filters(self, row: pd.Series) -> bool:
        """Verifica filtros obligatorios institucionales"""
        # Precio debe estar por encima de EMA200
        if not row.get('price_above_ema200', False):
            return False

        # Si se requiere alineación EMA, verificar
        if self.ema_alignment_required and row.get('ema_alignment', 0) != 1:
            return False

        # Si se usa IFVG, debe haber IFVG cercano
        if self.use_ifvg and not row.get('ifvg_bullish_nearby', False):
            return False

        return True

    def _calculate_signal_scores(self, row: pd.Series) -> Tuple[int, int]:
        """Calcula scores de compra y venta"""
        buy_signals = 0
        sell_signals = 0

        # 1. MACD
        macd = row.get('MACD', 0)
        if macd > self.macd_threshold:
            buy_signals += 1
        elif macd < -self.macd_threshold:
            sell_signals += 1

        # 2. Volumen
        volume_ratio = row.get('Volume_ratio', 1)
        if volume_ratio > 1.5:
            buy_signals += 1
        elif volume_ratio < 0.7:
            sell_signals += 1

        # 3. Squeeze Momentum
        squeeze_momentum = row.get('squeeze_momentum', 0)
        squeeze_delta = row.get('squeeze_momentum_delta', 0)
        if squeeze_momentum > self.squeeze_threshold:
            buy_signals += 1
        elif squeeze_momentum < -self.squeeze_threshold:
            sell_signals += 1

        # 4. IFVG
        if row.get('ifvg_bullish_nearby', False):
            buy_signals += 1
        elif row.get('ifvg_bearish_nearby', False):
            sell_signals += 1

        # 5. EMA Alignment
        ema_alignment = row.get('ema_alignment', 0)
        if ema_alignment == 1:
            buy_signals += 1
        elif ema_alignment == -1:
            sell_signals += 1

        # 6. Momentum (RSI + Squeeze delta)
        rsi = row.get('RSI', 50)
        momentum_score = 0
        if rsi < self.rsi_buy:
            momentum_score += 1
        elif rsi > self.rsi_sell:
            momentum_score -= 1

        if squeeze_delta > 0.2:
            momentum_score += 1
        elif squeeze_delta < -0.2:
            momentum_score -= 1

        if momentum_score > 0:
            buy_signals += 1
        elif momentum_score < 0:
            sell_signals += 1

        # 7. Trend Strength (EMA slope)
        ema22_slope = row.get('EMA_22_slope', 0)
        if ema22_slope > 0.001:
            buy_signals += 1
        elif ema22_slope < -0.001:
            sell_signals += 1

        return buy_signals, sell_signals

    def _get_buy_reasons(self, row: pd.Series) -> str:
        """Genera explicación detallada de señales de compra"""
        reasons = []

        if row.get('MACD', 0) > self.macd_threshold:
            reasons.append(".3f")
        if row.get('Volume_ratio', 1) > 1.5:
            reasons.append(".1f")
        if row.get('squeeze_momentum', 0) > self.squeeze_threshold:
            reasons.append(".2f")
        if row.get('ifvg_bullish_nearby', False):
            reasons.append("IFVG bullish cercano")
        if row.get('ema_alignment', 0) == 1:
            reasons.append("EMAs alineadas bullish")
        if row.get('squeeze_momentum_delta', 0) > 0.2:
            reasons.append(".2f")

        return "; ".join(reasons[:3])  # Máximo 3 razones

    def _get_sell_reasons(self, row: pd.Series) -> str:
        """Genera explicación detallada de señales de venta"""
        reasons = []

        if row.get('MACD', 0) < -self.macd_threshold:
            reasons.append(".3f")
        if row.get('Volume_ratio', 1) < 0.7:
            reasons.append("Volumen bajo")
        if row.get('squeeze_momentum', 0) < -self.squeeze_threshold:
            reasons.append(".2f")
        if row.get('ifvg_bearish_nearby', False):
            reasons.append("IFVG bearish cercano")
        if row.get('ema_alignment', 0) == -1:
            reasons.append("EMAs alineadas bearish")
        if row.get('squeeze_momentum_delta', 0) < -0.2:
            reasons.append(".2f")

        return "; ".join(reasons[:3])  # Máximo 3 razones


class GATrainer:
    """Entrenador de algoritmos genéticos para estrategias de trading"""

    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.data = None
        self.best_fitness_history = []

    def load_data(self, filepath: str):
        """Carga datos con indicadores"""
        print(f"Cargando datos desde {filepath}")
        self.data = pd.read_csv(filepath)

        # Añadir indicadores básicos
        self.data = add_all_indicators(self.data)

        # Añadir indicadores avanzados
        self.data = add_all_advanced_indicators(self.data)

        print(f"✓ Datos cargados: {len(self.data)} filas, {len(self.data.columns)} columnas")
        return self.data

    def fitness_function(self, genes: List[float]) -> float:
        """Función de fitness para evaluar estrategias GA"""
        try:
            strategy = GAStrategy(genes)
            return self._evaluate_strategy(strategy)
        except Exception as e:
            print(f"Error en fitness: {e}")
            return -1000  # Penalización por errores

    def _evaluate_strategy(self, strategy: GAStrategy) -> float:
        """Evalúa rendimiento de una estrategia"""
        capital = 10000
        position = 0
        entry_price = 0
        trades = []

        for idx, row in self.data.iterrows():
            decision, reason = strategy.decide(row, position)

            if decision == "BUY" and position == 0:
                # Calcular cantidad de acciones
                shares = int(capital * 0.95 / row['Close'])  # Usar 95% del capital
                if shares > 0:
                    position = shares
                    entry_price = row['Close']
                    capital -= shares * row['Close']

            elif decision == "SELL" and position > 0:
                # Vender posición
                capital += position * row['Close']
                pnl = (row['Close'] - entry_price) / entry_price
                trades.append(pnl)
                position = 0
                entry_price = 0

        # Calcular métricas finales
        total_return = (capital - 10000) / 10000 * 100
        num_trades = len(trades)

        if num_trades == 0:
            return -50  # Penalizar estrategias sin trades

        win_rate = sum(1 for t in trades if t > 0) / num_trades
        avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
        avg_loss = abs(np.mean([t for t in trades if t < 0])) if any(t < 0 for t in trades) else 0

        # Fitness: Retorno + Win Rate bonus - Drawdown penalty
        fitness = total_return + (win_rate * 20)  # Bonus por win rate

        # Penalizar alta volatilidad de retornos
        if num_trades > 1:
            returns_std = np.std(trades)
            fitness -= returns_std * 50

        # Bonus por profit factor
        if avg_loss > 0:
            profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss)
            fitness += min(profit_factor * 5, 20)  # Máximo bonus de 20

        return fitness

    def initialize_population(self) -> List[List[float]]:
        """Inicializa población aleatoria"""
        population = []
        for _ in range(self.population_size):
            genes = []
            # RSI thresholds
            genes.append(random.uniform(20, 40))  # RSI buy
            genes.append(random.uniform(60, 80))  # RSI sell
            # MACD threshold
            genes.append(random.uniform(0, 2))
            # Squeeze threshold
            genes.append(random.uniform(1, 3))
            # Boolean flags
            genes.append(random.uniform(0, 1))  # Use IFVG
            genes.append(random.uniform(0, 1))  # EMA alignment required
            # Risk management
            genes.append(random.uniform(1, 4))  # Stop loss %
            genes.append(random.uniform(2, 8))  # Take profit %
            # Weights (7) - one for each signal type
            for _ in range(7):
                genes.append(random.uniform(0, 1))

            population.append(genes)

        return population

    def select_parents(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        """Selección por torneo"""
        selected = []
        for _ in range(self.population_size):
            # Torneo de 3 individuos
            tournament = random.sample(list(zip(population, fitness_scores)), 3)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Crossover de un punto"""
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, individual: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Mutación gaussiana"""
        mutated = []
        for i, gene in enumerate(individual):
            if random.random() < mutation_rate:
                # Mutación gaussiana
                mutation = random.gauss(0, 0.1)
                new_gene = gene + mutation
                # Clamp a rangos razonables
                if i < 2:  # RSI
                    new_gene = max(10, min(90, new_gene))
                elif i in [4, 5]:  # Boolean
                    new_gene = max(0, min(1, new_gene))
                elif i == 6:  # Stop loss
                    new_gene = max(0.5, min(5, new_gene))
                elif i == 7:  # Take profit
                    new_gene = max(1, min(10, new_gene))
                else:  # Otros
                    new_gene = max(0, min(5, new_gene))
                mutated.append(new_gene)
            else:
                mutated.append(gene)
        return mutated

    def evolve_population(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        """Evoluciona la población"""
        # Seleccionar padres
        parents = self.select_parents(population, fitness_scores)

        # Crear nueva generación
        new_population = []

        # Elitismo: mantener el mejor individuo
        best_idx = fitness_scores.index(max(fitness_scores))
        new_population.append(population[best_idx])

        # Generar resto mediante crossover y mutación
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.extend([child1, child2])

        return new_population[:self.population_size]

    def train(self, data_path: str, save_path: str = None) -> Dict:
        """Entrena el algoritmo genético"""
        print("🧬 INICIANDO ENTRENAMIENTO GA - ESTRATEGIA AVANZADA")
        print(f"Población: {self.population_size} individuos")
        print(f"Generaciones: {self.generations}")

        # Cargar datos
        self.load_data(data_path)

        # Inicializar población
        population = self.initialize_population()
        best_fitness = -1000
        best_individual = None

        print("\n🚀 Evolución comenzando...")

        for generation in range(self.generations):
            # Evaluar fitness
            fitness_scores = [self.fitness_function(individual) for individual in population]

            # Actualizar mejor individuo
            current_best_fitness = max(fitness_scores)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[fitness_scores.index(current_best_fitness)]

            self.best_fitness_history.append(best_fitness)

            # Mostrar progreso cada 10 generaciones
            if (generation + 1) % 10 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                print(f"Gen {generation+1:3d}: Mejor Fitness: {best_fitness:6.2f} | Promedio: {avg_fitness:6.2f}")

            # Evolucionar (excepto en la última generación)
            if generation < self.generations - 1:
                population = self.evolve_population(population, fitness_scores)

        print("\n✅ Entrenamiento completado!")
        print(f"Mejor Fitness: {best_fitness:.2f}")

        # Verificar que tenemos un individuo válido
        if best_individual is None:
            print("⚠️  No se encontró ningún individuo válido. Usando individuo por defecto.")
            best_individual = self.initialize_population()[0]  # Usar primer individuo de nueva población

        # Crear y mostrar mejor estrategia
        best_strategy = GAStrategy(best_individual)
        print("\n📊 MEJOR ESTRATEGIA ENCONTRADA:")
        print(f"  RSI Buy/Sell: {best_strategy.rsi_buy:.1f}/{best_strategy.rsi_sell:.1f}")
        print(f"  MACD Threshold: {best_strategy.macd_threshold:.3f}")
        print(f"  Squeeze Threshold: {best_strategy.squeeze_threshold:.2f}")
        print(f"  Use IFVG: {best_strategy.use_ifvg}")
        print(f"  EMA Alignment Required: {best_strategy.ema_alignment_required}")
        print(f"  Stop Loss/Take Profit: {best_strategy.stop_loss:.1f}%/{best_strategy.take_profit:.1f}%")

        # Guardar modelo si se especifica
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model_data = {
                'genes': best_individual,
                'fitness': best_fitness,
                'strategy': best_strategy,
                'training_date': datetime.now().isoformat(),
                'generations': self.generations,
                'population_size': self.population_size
            }
            joblib.dump(model_data, save_path)
            print(f"💾 Modelo guardado en: {save_path}")

        return {
            'best_genes': best_individual,
            'best_fitness': best_fitness,
            'strategy': best_strategy,
            'fitness_history': self.best_fitness_history
        }


if __name__ == "__main__":
    # Entrenar modelo GA avanzado
    trainer = GATrainer(population_size=50, generations=100)

    data_path = "data/processed/SPY_with_indicators.csv"
    save_path = "models/ga_advanced_strategy.pkl"

    results = trainer.train(data_path, save_path)

    print("\n🎯 ENTRENAMIENTO COMPLETADO")
    print(f"Fitness final: {results['best_fitness']:.2f}")
    print(f"Modelo guardado: {save_path}")