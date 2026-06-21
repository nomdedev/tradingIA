"""
Cargador de Reglas Declarativas (YAML) para el Council.
Convierte definiciones YAML en objetos de regla evaluables.
"""

import yaml
import os
from typing import List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class Rule:
    def __init__(self, definition: Dict[str, Any]):
        self.id = definition.get("id")
        self.type = definition.get("type")
        self.description = definition.get("description")
        self.metric = definition.get("metric")
        self.operator = definition.get("operator")
        self.value = definition.get("value")
        self.severity = definition.get("severity", "info")
        self.action = definition.get("action", "log")
        self.expert = definition.get("expert", "Trend Master")  # Default expert

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa la regla contra el contexto proporcionado.
        """
        metric_value = context.get(self.metric)

        if metric_value is None:
            return {
                "rule_id": self.id,
                "passed": False,
                "reason": f"Metric '{self.metric}' not found in context",
                "severity": self.severity,
                "action": self.action,
                "expert": self.expert,
            }

        passed = False
        try:
            if self.operator == ">":
                passed = metric_value > self.value
            elif self.operator == ">=":
                passed = metric_value >= self.value
            elif self.operator == "<":
                passed = metric_value < self.value
            elif self.operator == "<=":
                passed = metric_value <= self.value
            elif self.operator == "==":
                passed = metric_value == self.value
            elif self.operator == "!=":
                passed = metric_value != self.value
            else:
                return {
                    "rule_id": self.id,
                    "passed": False,
                    "reason": f"Unknown operator '{self.operator}'",
                    "severity": self.severity,
                    "action": self.action,
                    "expert": self.expert,
                }
        except Exception as e:
            return {
                "rule_id": self.id,
                "passed": False,
                "reason": f"Evaluation error: {str(e)}",
                "severity": self.severity,
                "action": self.action,
                "expert": self.expert,
            }

        return {
            "rule_id": self.id,
            "passed": passed,
            "metric_value": metric_value,
            "threshold": self.value,
            "severity": self.severity,
            "action": self.action if not passed else "none",
            "expert": self.expert,
        }


class RulesLoader:
    def __init__(self, rules_dir: str):
        self.rules_dir = rules_dir
        self.rules: Dict[str, List[Rule]] = {}

    def load_rules(self):
        """Carga todas las reglas YAML del directorio."""
        if not os.path.exists(self.rules_dir):
            logger.warning(f"Rules directory {self.rules_dir} does not exist.")
            return

        for filename in os.listdir(self.rules_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                category = os.path.splitext(filename)[0]
                filepath = os.path.join(self.rules_dir, filename)

                try:
                    with open(filepath, "r") as f:
                        data = yaml.safe_load(f)
                        if data and "rules" in data:
                            self.rules[category] = [Rule(r) for r in data["rules"]]
                            logger.info(f"Loaded {len(self.rules[category])} rules from {filename}")
                except Exception as e:
                    logger.error(f"Failed to load rules from {filename}: {e}")

    def get_rules(self, category: str) -> List[Rule]:
        return self.rules.get(category, [])

    def evaluate_category(self, category: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa todas las reglas de una categoría."""
        rules = self.get_rules(category)
        results = [rule.evaluate(context) for rule in rules]

        # Resumen
        failed_critical = any(r["passed"] is False and r["severity"] == "critical" for r in results)
        failed_warning = any(r["passed"] is False and r["severity"] == "warning" for r in results)

        status = "PASS"
        if failed_critical:
            status = "REJECT"
        elif failed_warning:
            status = "WARNING"

        return {"category": category, "status": status, "results": results}
