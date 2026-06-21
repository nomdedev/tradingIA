"""Módulo para orquestar reglas y expertos en decisiones de trading.

API:
- Council.register_expert(name, role, weight)
- Council.add_rule(rule_id, func, description)
- Council.load_rules_from_yaml(rules_dir)
- Council.decide(context) -> dict (decision, aggregate_score, details)
- Council.validate_context(context, category) -> bool

Las reglas pueden ser:
1. Callables (Python functions)
2. Declarativas (YAML loaded via RulesLoader)
"""

import math
from typing import Callable, Dict, Any, Optional, List, Tuple
import os
from core.rules_loader import RulesLoader

# Constantes para nombres de expertos
EXPERT_RISK_WARDEN = "Risk Warden"
EXPERT_TREND_MASTER = "Trend Master"
EXPERT_DATA_ORACLE = "Data Oracle"
EXPERT_ARCHITECT_PRIME = "Architect Prime"
EXPERT_SENTIMENT_SEER = "Sentiment Seer"

# Umbrales de certificación
CERT_APPROVED_THRESHOLD = 0.6
CERT_REJECTED_THRESHOLD = 0.4


class Council:
    def __init__(self, rules_dir: str = None):
        self.experts: Dict[str, Dict[str, Any]] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.rules_loader: Optional[RulesLoader] = None
        self.certified_strategies: Dict[str, Dict[str, Any]] = {}
        self.known_patterns: list = []

        if rules_dir:
            self.load_rules_from_yaml(rules_dir)

    def certify_strategy(self, strategy_id: str, stability_score: float, details: Dict = None) -> None:
        """
        Registra la certificación de una estrategia basada en WFA.
        
        Args:
            strategy_id: ID único de la estrategia.
            stability_score: Puntaje de estabilidad (0.0 a 1.0) del WFA.
            details: Detalles adicionales del análisis.
        """
        status = self._determine_certification_status(stability_score)
        self.certified_strategies[strategy_id] = {
            "score": stability_score,
            "status": status,
            "details": details or {}
        }

    def _determine_certification_status(self, stability_score: float) -> str:
        """Determina el estado de certificación basado en el score."""
        if stability_score >= CERT_APPROVED_THRESHOLD:
            return "APPROVED"
        elif stability_score < CERT_REJECTED_THRESHOLD:
            return "REJECTED"
        return "WARNING"

    def _check_strategy_certification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Regla interna: Verifica si la estrategia está certificada."""
        strat_id = context.get("strategy_id")
        if not strat_id:
            return {"signal": 0, "score": 0.0, "details": "No strategy_id provided"}
            
        cert = self.certified_strategies.get(strat_id)
        if not cert:
            # Si no está certificada, penalizamos levemente pero no vetamos (podría ser nueva)
            # O mejor, Trend Master debería ser escéptico.
            return {"signal": 0, "score": -0.2, "details": "Strategy NOT certified"}
            
        if cert["status"] == "APPROVED":
            return {"signal": 1, "score": cert["score"], "details": f"Certified (Score: {cert['score']:.2f})"}
        elif cert["status"] == "WARNING":
            return {"signal": 0, "score": 0.0, "details": f"Certified with WARNING (Score: {cert['score']:.2f})"}
        else:
            return {"signal": -1, "score": -1.0, "details": f"Strategy REJECTED by WFA (Score: {cert['score']:.2f})"}

    def register_expert(
        self, name: str, role: str, domain: str = "trading", weight: float = 1.0, notes: str = ""
    ) -> None:
        """
        Registra un experto en el Consejo.

        Args:
            name: Nombre del experto/agente.
            role: Rol (ej. 'Risk Manager', 'Trend Analyst').
            domain: Dominio de expertise ('trading', 'system', 'data', 'risk').
            weight: Peso de su voto (0.0 a 1.0+).
            notes: Notas adicionales.
        """
        self.experts[name] = {"role": role, "domain": domain, "weight": float(weight), "notes": notes}

    def register_standard_experts(self) -> None:
        """Registra los expertos estándar definidos en el protocolo."""
        self.register_expert(EXPERT_RISK_WARDEN, "Security Officer", "risk", 2.5, "VETO ABSOLUTO")
        self.register_expert(EXPERT_TREND_MASTER, "Strategy Lead", "trading", 2.0, "Voto Calificado")
        self.register_expert(EXPERT_DATA_ORACLE, "Data Engineer", "data", 1.0, "Veto Técnico")
        self.register_expert(EXPERT_ARCHITECT_PRIME, "System Architect", "system", 1.0, "Veto Técnico")
        self.register_expert(EXPERT_SENTIMENT_SEER, "Analyst", "sentiment", 1.0, "Voto Consultivo")
        
        # Register internal rules
        self.add_rule("wfa_certification", self._check_strategy_certification, "Verifica certificación WFA", expert=EXPERT_TREND_MASTER)
        self.add_rule("pattern_confluence", self._check_pattern_confluence, "Verifica confluencia de patrones", expert=EXPERT_TREND_MASTER)
        
        # ÁREA 7: Register data quality rule for Data Oracle
        self.add_rule("data_quality", self._check_data_quality, "Verifica calidad de datos", expert=EXPERT_DATA_ORACLE)

    def _check_data_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        ÁREA 7: Regla del Data Oracle para verificar calidad de datos.
        
        El Data Oracle puede vetar trades si:
        - Los datos tienen gaps significativos
        - El volumen es 0 o anómalo
        - Hay errores críticos de validación
        """
        data_quality = context.get("data_quality", {})
        
        if not data_quality:
            # No data quality info provided - neutral
            return {"signal": 0, "score": 0.0, "details": "No data quality info available"}
        
        # Check for critical issues
        if data_quality.get("has_gaps", False):
            return {"signal": -1, "score": -0.8, "details": "Data has time gaps - unreliable for trading"}
        
        if not data_quality.get("volume_ok", True):
            return {"signal": -1, "score": -0.6, "details": "Zero or invalid volume detected"}
        
        # Check validation score if available
        quality_score = data_quality.get("score", 0.5)
        
        if quality_score < 0.3:
            return {"signal": -1, "score": -1.0, "details": f"Data quality score too low: {quality_score:.2f}"}
        
        if quality_score < 0.5:
            return {"signal": 0, "score": -0.3, "details": f"Data quality below average: {quality_score:.2f}"}
        
        # Check for specific issues
        issues = data_quality.get("issues", [])
        if issues:
            # Count severity of issues
            critical_count = sum(1 for i in issues if "CRITICAL" in i.upper())
            error_count = sum(1 for i in issues if "ERROR" in i.upper())
            
            if critical_count > 0:
                return {"signal": -1, "score": -1.0, "details": f"Critical data issues: {issues[:3]}"}
            
            if error_count > 2:
                return {"signal": -1, "score": -0.7, "details": f"Multiple data errors: {error_count}"}
        
        # Data looks good
        if quality_score >= 0.8:
            return {"signal": 1, "score": 0.5, "details": f"High quality data: {quality_score:.2f}"}
        
        return {"signal": 1, "score": 0.2, "details": f"Data quality acceptable: {quality_score:.2f}"}

    def register_pattern(self, pattern: Dict[str, Any]) -> None:
        """Registra un patrón descubierto (ej. de Tab 7)."""
        self.known_patterns.append(pattern)

    def _check_pattern_confluence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Regla interna: Verifica si el contexto actual coincide con algún patrón conocido."""
        # El contexto debe incluir 'active_patterns' detectados por la estrategia
        active_patterns = context.get("active_patterns", [])
        if not active_patterns:
            return {"signal": 0, "score": 0.0, "details": "No active patterns"}
            
        score_sum = 0.0
        matched = []
        
        for p_name in active_patterns:
            # Buscar si este patrón está en nuestra lista de alta probabilidad
            known = next((p for p in self.known_patterns if p.get('pattern_name') == p_name), None)
            if known:
                # Boost score basado en win rate
                wr = known.get('win_rate', 0.5)
                if wr > 0.6:
                    # Bonus: 0.6 -> 0.2, 0.8 -> 0.6, 1.0 -> 1.0
                    bonus = (wr - 0.5) * 2
                    score_sum += bonus
                    matched.append(f"{p_name} ({wr:.0%})")
                    
        if score_sum > 0:
            return {"signal": 1, "score": min(score_sum, 1.0), "details": f"Pattern Confluence: {', '.join(matched)}"}
            
        return {"signal": 0, "score": 0.0, "details": "No high-probability patterns matched"}

    def get_experts_by_domain(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """Retorna los expertos de un dominio específico."""
        return {k: v for k, v in self.experts.items() if v.get("domain") == domain}

    def add_rule(
        self,
        rule_id: str,
        func: Callable[[Dict[str, Any]], Dict[str, Any]],
        description: str = "",
        expert: str = "Trend Master",
    ) -> None:
        """Registra una regla ejecutable (Python function)."""
        self.rules[rule_id] = {"func": func, "description": description, "type": "executable", "expert": expert}

    def load_rules_from_yaml(self, rules_dir: str) -> None:
        """Carga reglas declarativas desde un directorio YAML."""
        self.rules_loader = RulesLoader(rules_dir)
        self.rules_loader.load_rules()

    def validate_context(self, context: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        Valida el contexto contra un conjunto de reglas declarativas (ej. 'data_quality').
        Retorna el resultado de la evaluación.
        """
        if not self.rules_loader:
            return {"status": "PASS", "reason": "No rules loader configured"}

        return self.rules_loader.evaluate_category(category, context)

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa todas las reglas ejecutables registradas."""
        results: Dict[str, Any] = {}
        for rid, meta in self.rules.items():
            if meta.get("type") == "executable":
                try:
                    out = meta["func"](context)
                except Exception as e:
                    out = {"error": str(e)}
                results[rid] = out
        return results

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma una decisión de trading basada en el Protocolo de Interacción del Consejo.

        Fases:
        1. Recolección de Evidencia (Ejecución de reglas)
        2. Formación de Opinión (Voto por experto)
        3. Ronda de Vetos (Risk/System/Data)
        4. Consenso Ponderado (Score final)
        """
        # Si no hay expertos registrados, registrar los estándar
        if not self.experts:
            self.register_standard_experts()

        # --- Fase 1: Recolección de Evidencia ---
        executable_results = self.evaluate(context)
        declarative_results = self._evaluate_declarative_rules(context)

        # --- Fase 2: Formación de Opinión del Experto ---
        expert_evidence = self._gather_expert_evidence(executable_results, declarative_results)
        expert_votes, expert_reasons = self._calculate_expert_votes(expert_evidence)

        # --- Fase 3: Ronda de Vetos ---
        veto_result = self._check_vetos(expert_votes, expert_reasons)
        if veto_result:
            return veto_result

        # --- Fase 4: Consenso Ponderado ---
        return self._calculate_consensus(expert_votes)

    def _evaluate_declarative_rules(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa reglas declarativas si existen."""
        declarative_results = {}
        if not self.rules_loader:
            return declarative_results
            
        # Mapeo de categorías a expertos por defecto
        category_map = {
            "risk_limits": EXPERT_RISK_WARDEN,
            "data_quality": EXPERT_DATA_ORACLE,
            "system_health": EXPERT_ARCHITECT_PRIME,
            "strategies": EXPERT_TREND_MASTER,
        }
        
        for cat, expert in category_map.items():
            res = self.rules_loader.evaluate_category(cat, context)
            if res["status"] in ["REJECT", "FAIL"]:
                declarative_results[f"decl_{cat}"] = {
                    "score": -1.0, "signal": -1, "expert": expert, "details": res
                }
            elif res["status"] == "PASS":
                declarative_results[f"decl_{cat}"] = {
                    "score": 1.0, "signal": 1, "expert": expert, "details": res
                }
        
        return declarative_results

    def _gather_expert_evidence(
        self, 
        executable_results: Dict, 
        declarative_results: Dict
    ) -> Dict[str, List]:
        """Agrupa evidencia por experto."""
        expert_evidence = {name: [] for name in self.experts}
        
        # Procesar ejecutables
        for rid, res in executable_results.items():
            expert = self.rules.get(rid, {}).get("expert", EXPERT_TREND_MASTER)
            if expert not in expert_evidence:
                expert_evidence[expert] = []
            expert_evidence[expert].append(res)
        
        # Procesar declarativas
        for rid, res in declarative_results.items():
            expert = res.get("expert", EXPERT_TREND_MASTER)
            if expert not in expert_evidence:
                expert_evidence[expert] = []
            expert_evidence[expert].append(res)
        
        return expert_evidence

    def _calculate_expert_votes(
        self, 
        expert_evidence: Dict[str, List]
    ) -> Tuple[Dict[str, int], Dict[str, List]]:
        """Calcula el voto de cada experto basado en su evidencia."""
        expert_votes = {}
        expert_reasons = {}
        
        for name, evidence in expert_evidence.items():
            if not evidence:
                expert_votes[name] = 0  # ABSTAIN
                continue
            
            vote, reasons = self._calculate_single_vote(evidence)
            expert_votes[name] = vote
            if vote == -1:
                expert_reasons[name] = reasons
        
        return expert_votes, expert_reasons

    def _calculate_single_vote(self, evidence: List) -> Tuple[int, List]:
        """Calcula el voto para un experto basado en su evidencia."""
        reasons = []
        has_reject = False
        score_sum = 0.0
        
        for item in evidence:
            sig = item.get("signal", 0)
            score = item.get("score", 0)
            
            if sig == -1 or (score is not None and score < 0):
                has_reject = True
                reasons.append(item.get("details", "Rule failed"))
            
            if score is not None:
                try:
                    score_sum += float(score)
                except (TypeError, ValueError):
                    pass  # Score inválido, no se suma
        
        if has_reject:
            return -1, reasons
        elif score_sum > 0:
            return 1, reasons
        return 0, reasons

    def _check_vetos(
        self, 
        expert_votes: Dict[str, int], 
        expert_reasons: Dict[str, List]
    ) -> Optional[Dict[str, Any]]:
        """Verifica si algún experto con poder de veto ha rechazado."""
        # Risk Warden Veto
        if expert_votes.get(EXPERT_RISK_WARDEN, 0) == -1:
            return self._create_veto_response(
                EXPERT_RISK_WARDEN, expert_votes, expert_reasons
            )
        
        # Technical Vetos (Data/System)
        for tech_expert in [EXPERT_DATA_ORACLE, EXPERT_ARCHITECT_PRIME]:
            if expert_votes.get(tech_expert, 0) == -1:
                return self._create_veto_response(
                    tech_expert, expert_votes, expert_reasons
                )
        
        return None

    def _create_veto_response(
        self, 
        expert: str, 
        expert_votes: Dict, 
        expert_reasons: Dict
    ) -> Dict[str, Any]:
        """Crea respuesta de veto."""
        return {
            "decision": 0,
            "aggregate_score": -1.0,
            "reason": f"{expert} VETO: {expert_reasons.get(expert)}",
            "expert_votes": expert_votes,
            "phase": "VETO",
        }

    def _calculate_consensus(self, expert_votes: Dict[str, int]) -> Dict[str, Any]:
        """Calcula el consenso ponderado final."""
        total_score = 0.0
        total_weight = 0.0
        
        for name, vote in expert_votes.items():
            if vote != 0:  # Excluir abstenciones
                weight = self.experts.get(name, {}).get("weight", 1.0)
                total_score += vote * weight
                total_weight += weight
        
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        decision = self._determine_decision(final_score, total_weight, expert_votes)
        
        return {
            "decision": decision,
            "aggregate_score": final_score,
            "expert_votes": expert_votes,
            "total_weight": total_weight,
            "phase": "CONSENSUS",
        }

    def _determine_decision(
        self, 
        final_score: float, 
        total_weight: float, 
        expert_votes: Dict
    ) -> int:
        """Determina la decisión final basada en el score."""
        if final_score > 0.0:
            return 1  # BUY
        elif final_score < 0.0:
            return -1  # REJECT
        
        # Manejo de empate técnico (Risk Warden rompe empate)
        if math.isclose(final_score, 0.0, abs_tol=1e-9) and total_weight > 0:
            if expert_votes.get(EXPERT_RISK_WARDEN, 0) < 0:
                return 0
        
        return 0
