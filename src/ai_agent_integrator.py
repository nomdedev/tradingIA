"""
AI Agent Integration Module
============================

Integración con agentes IA para análisis automático de:
- Estrategias de trading
- Resultados de backtesting
- Optimización de parámetros
- Validación matemática
- Sugerencias de mejora
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import requests


class AIAgentIntegrator:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.analysis_history = []

    def is_enabled(self) -> bool:
        """Check if AI integration is enabled"""
        return self.config_manager.is_ai_enabled()

    def analyze_backtest_results(self, strategy_name: str, params: Dict,
                                 metrics: Dict, trades: List[Dict]) -> Optional[Dict]:
        """
        Analyze backtest results using configured AI agent

        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters used
            metrics: Performance metrics
            trades: List of trades executed

        Returns:
            Analysis report with insights and suggestions
        """
        if not self.is_enabled():
            self.logger.info("AI analysis disabled")
            return None

        try:
            active_agent = self.config_manager.get_active_agent()
            if not active_agent:
                return None

            # Prepare analysis prompt
            prompt = self._prepare_backtest_analysis_prompt(
                strategy_name, params, metrics, trades
            )

            # Get analysis from agent
            if active_agent == 'copilot':
                result = self._analyze_with_copilot(prompt)
            elif active_agent == 'claude':
                result = self._analyze_with_claude(prompt)
            elif active_agent == 'chatgpt':
                result = self._analyze_with_chatgpt(prompt)
            else:
                result = self._analyze_with_custom_agent(prompt, active_agent)

            # Save analysis to history
            self._save_analysis(strategy_name, result)

            return result

        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return None

    def analyze_strategy_code(self, strategy_name: str, code: str) -> Optional[Dict]:
        """
        Analyze strategy code for potential issues and improvements

        Args:
            strategy_name: Name of the strategy
            code: Strategy code to analyze

        Returns:
            Code analysis with suggestions
        """
        if not self.is_enabled():
            return None

        try:
            active_agent = self.config_manager.get_active_agent()

            prompt = self._prepare_code_analysis_prompt(strategy_name, code)

            if active_agent == 'copilot':
                result = self._analyze_with_copilot(prompt)
            elif active_agent == 'claude':
                result = self._analyze_with_claude(prompt)
            elif active_agent == 'chatgpt':
                result = self._analyze_with_chatgpt(prompt)
            else:
                result = self._analyze_with_custom_agent(prompt, active_agent)

            return result

        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            return None

    def compare_strategies(self, comparisons: List[Dict]) -> Optional[Dict]:
        """
        Compare multiple strategy results and provide insights

        Args:
            comparisons: List of strategy results to compare

        Returns:
            Comparison analysis
        """
        if not self.is_enabled():
            return None

        try:
            active_agent = self.config_manager.get_active_agent()

            prompt = self._prepare_comparison_prompt(comparisons)

            if active_agent == 'copilot':
                result = self._analyze_with_copilot(prompt)
            elif active_agent == 'claude':
                result = self._analyze_with_claude(prompt)
            elif active_agent == 'chatgpt':
                result = self._analyze_with_chatgpt(prompt)
            else:
                result = self._analyze_with_custom_agent(prompt, active_agent)

            return result

        except Exception as e:
            self.logger.error(f"Strategy comparison failed: {e}")
            return None

    def suggest_parameter_optimization(self, strategy_name: str,
                                       current_params: Dict,
                                       performance_metrics: Dict) -> Optional[Dict]:
        """
        Get AI suggestions for parameter optimization

        Args:
            strategy_name: Name of the strategy
            current_params: Current parameter values
            performance_metrics: Current performance metrics

        Returns:
            Parameter optimization suggestions
        """
        if not self.is_enabled():
            return None

        try:
            active_agent = self.config_manager.get_active_agent()

            prompt = self._prepare_optimization_prompt(
                strategy_name, current_params, performance_metrics
            )

            if active_agent == 'copilot':
                result = self._analyze_with_copilot(prompt)
            elif active_agent == 'claude':
                result = self._analyze_with_claude(prompt)
            elif active_agent == 'chatgpt':
                result = self._analyze_with_chatgpt(prompt)
            else:
                result = self._analyze_with_custom_agent(prompt, active_agent)

            return result

        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            return None

    def validate_mathematical_model(self, model_description: str,
                                    formulas: List[str]) -> Optional[Dict]:
        """
        Validate mathematical correctness of trading models

        Args:
            model_description: Description of the trading model
            formulas: List of mathematical formulas used

        Returns:
            Mathematical validation report
        """
        if not self.is_enabled():
            return None

        options = self.config_manager.get_analysis_options()
        if not options.get('include_mathematical_validation', False):
            return None

        try:
            active_agent = self.config_manager.get_active_agent()

            prompt = f"""
Validate the mathematical correctness of this trading model:

Model Description:
{model_description}

Formulas:
{chr(10).join(f"- {formula}" for formula in formulas)}

Please check for:
1. Mathematical correctness
2. Edge cases and potential division by zero
3. Numerical stability issues
4. Statistical validity
5. Recommendations for improvements
"""

            if active_agent == 'claude':
                result = self._analyze_with_claude(prompt)
            elif active_agent == 'chatgpt':
                result = self._analyze_with_chatgpt(prompt)
            else:
                result = self._analyze_with_copilot(prompt)

            return result

        except Exception as e:
            self.logger.error(f"Mathematical validation failed: {e}")
            return None

    # ==================== Agent-Specific Methods ====================

    def _analyze_with_copilot(self, prompt: str) -> Dict:
        """
        Analyze using GitHub Copilot (via VSCode API)
        Note: This is a placeholder - actual implementation would use VSCode extension API
        """
        self.logger.info("Analysis with GitHub Copilot requested")

        return {
            'agent': 'copilot',
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'analysis': 'GitHub Copilot analysis - would integrate with VSCode extension API',
            'suggestions': [
                'Integration requires VSCode extension API',
                'Consider using Claude or ChatGPT for standalone analysis'
            ],
            'note': 'This is a placeholder for VSCode Copilot integration'
        }

    def _analyze_with_claude(self, prompt: str) -> Dict:
        """Analyze using Claude API"""
        try:
            agents_config = self.config_manager.config['ai_integration']['available_agents']
            claude_config = agents_config.get('claude', {})

            api_key = claude_config.get('api_key', '')
            if not api_key:
                return {
                    'status': 'error',
                    'message': 'Claude API key not configured'
                }

            # Decrypt API key if encrypted
            if self.config_manager.config['security_settings']['encrypt_credentials']:
                api_key = self.config_manager._decrypt(api_key)

            # Call Claude API
            headers = {
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            }

            data = {
                'model': 'claude-3-opus-20240229',
                'max_tokens': 4096,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ]
            }

            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    'agent': 'claude',
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'analysis': result['content'][0]['text'],
                    'model': result['model'],
                    'usage': result.get('usage', {})
                }
            else:
                return {
                    'status': 'error',
                    'message': f"Claude API error: {response.status_code}",
                    'details': response.text
                }

        except Exception as e:
            self.logger.error(f"Claude analysis failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _analyze_with_chatgpt(self, prompt: str) -> Dict:
        """Analyze using ChatGPT API"""
        try:
            agents_config = self.config_manager.config['ai_integration']['available_agents']
            chatgpt_config = agents_config.get('chatgpt', {})

            api_key = chatgpt_config.get('api_key', '')
            if not api_key:
                return {
                    'status': 'error',
                    'message': 'ChatGPT API key not configured'
                }

            # Decrypt API key if encrypted
            if self.config_manager.config['security_settings']['encrypt_credentials']:
                api_key = self.config_manager._decrypt(api_key)

            # Call OpenAI API
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            data = {'model': 'gpt-4-turbo-preview',
                    'messages': [{'role': 'system',
                                  'content': 'You are an expert quantitative trading analyst specialized in backtesting analysis and strategy optimization.'},
                                 {'role': 'user',
                                  'content': prompt}],
                    'temperature': 0.7,
                    'max_tokens': 4096}

            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    'agent': 'chatgpt',
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'analysis': result['choices'][0]['message']['content'],
                    'model': result['model'],
                    'usage': result.get('usage', {})
                }
            else:
                return {
                    'status': 'error',
                    'message': f"ChatGPT API error: {response.status_code}",
                    'details': response.text
                }

        except Exception as e:
            self.logger.error(f"ChatGPT analysis failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _analyze_with_custom_agent(self, prompt: str, agent_name: str) -> Dict:
        """Analyze using custom agent endpoint"""
        try:
            agents_config = self.config_manager.config['ai_integration']['available_agents']
            custom_config = agents_config.get(agent_name, {})

            endpoint = custom_config.get('endpoint', '')
            api_key = custom_config.get('api_key', '')

            if not endpoint:
                return {
                    'status': 'error',
                    'message': f'Custom agent {agent_name} endpoint not configured'
                }

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            data = {'prompt': prompt}

            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return {
                    'agent': agent_name,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'analysis': response.json().get('response', response.text)
                }
            else:
                return {
                    'status': 'error',
                    'message': f"Custom agent error: {response.status_code}"
                }

        except Exception as e:
            self.logger.error(f"Custom agent analysis failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    # ==================== Prompt Preparation ====================

    def _prepare_backtest_analysis_prompt(self, strategy_name: str, params: Dict,
                                          metrics: Dict, trades: List[Dict]) -> str:
        """Prepare prompt for backtest analysis"""
        options = self.config_manager.get_analysis_options()

        prompt = f"""
Analyze the following backtesting results for a trading strategy:

Strategy: {strategy_name}

Parameters:
{json.dumps(params, indent=2)}

Performance Metrics:
{json.dumps(metrics, indent=2)}

Number of Trades: {len(trades)}
Sample Trades (first 5):
{json.dumps(trades[:5], indent=2)}

"""

        if options.get('include_statistical_tests', True):
            prompt += "\nPlease include:\n"
            prompt += "1. Statistical significance assessment\n"
            prompt += "2. Risk-adjusted performance evaluation\n"

        if options.get('include_risk_assessment', True):
            prompt += "3. Risk profile analysis\n"
            prompt += "4. Drawdown analysis and recovery periods\n"

        if options.get('include_improvement_suggestions', True):
            prompt += "5. Specific suggestions for improvement\n"
            prompt += "6. Parameter optimization recommendations\n"

        if options.get('include_mathematical_validation', True):
            prompt += "7. Mathematical validity of the approach\n"

        return prompt

    def _prepare_code_analysis_prompt(self, strategy_name: str, code: str) -> str:
        """Prepare prompt for code analysis"""
        return f"""
Analyze the following trading strategy code:

Strategy Name: {strategy_name}

Code:
```python
{code}
```

Please provide:
1. Code quality assessment
2. Potential bugs or edge cases
3. Performance optimization suggestions
4. Best practices recommendations
5. Security considerations
6. Testing recommendations
"""

    def _prepare_comparison_prompt(self, comparisons: List[Dict]) -> str:
        """Prepare prompt for strategy comparison"""
        return f"""
Compare the following trading strategies:

{json.dumps(comparisons, indent=2)}

Please provide:
1. Comparative performance analysis
2. Risk-return profile comparison
3. Strengths and weaknesses of each strategy
4. Recommendation on which strategy to use
5. Suggestions for combining strategies
"""

    def _prepare_optimization_prompt(self, strategy_name: str,
                                     current_params: Dict,
                                     performance_metrics: Dict) -> str:
        """Prepare prompt for parameter optimization"""
        return f"""
Suggest parameter optimization for the following strategy:

Strategy: {strategy_name}

Current Parameters:
{json.dumps(current_params, indent=2)}

Current Performance:
{json.dumps(performance_metrics, indent=2)}

Please provide:
1. Analysis of current parameter settings
2. Specific parameter value suggestions
3. Reasoning for each suggestion
4. Expected impact on performance
5. Risk considerations
"""

    def _save_analysis(self, strategy_name: str, analysis: Dict):
        """Save analysis to history"""
        try:
            analysis_record = {
                'strategy': strategy_name,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis
            }

            self.analysis_history.append(analysis_record)

            # Save to file
            history_file = Path('results/ai_analysis_history.json')
            history_file.parent.mkdir(exist_ok=True)

            with open(history_file, 'w') as f:
                json.dump(self.analysis_history, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save analysis: {e}")

    def get_analysis_history(self, strategy_name: Optional[str] = None,
                             limit: int = 10) -> List[Dict]:
        """Get analysis history"""
        if strategy_name:
            return [
                record for record in self.analysis_history[-limit:]
                if record['strategy'] == strategy_name
            ]
        return self.analysis_history[-limit:]
