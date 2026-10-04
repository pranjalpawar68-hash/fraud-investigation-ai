"""
Fraud Investigation Orchestrator
Routes cases through the agent pipeline with conditional logic.
"""

import time
import random
from datetime import datetime
from agents.transaction_agent import TransactionAnalysisAgent
from agents.customer_agent import CustomerBehaviourAgent
from agents.risk_agent import RiskPolicyAgent
from agents.recommendation_agent import RecommendationAgent


class FraudOrchestrator:
    """Central orchestrator that routes fraud cases through the agent pipeline."""

    def __init__(self, model_name="Qwen-2.5-72B"):
        self.model_name = model_name
        self.transaction_agent = TransactionAnalysisAgent(model_name)
        self.customer_agent = CustomerBehaviourAgent(model_name)
        self.risk_agent = RiskPolicyAgent(model_name)
        self.recommendation_agent = RecommendationAgent(model_name)
        self.pipeline_log = []

    def _log(self, agent_name, status, message, duration_ms=0):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "status": status,
            "message": message,
            "duration_ms": duration_ms,
            "model": self.model_name,
        }
        self.pipeline_log.append(entry)
        return entry

    def run_investigation(self, case_data, progress_callback=None):
        """Run the full investigation pipeline on a case."""
        self.pipeline_log = []
        results = {}
        start_time = time.time()

        # Step 1: Transaction Analysis
        self._log("Orchestrator", "routing", "Routing to Transaction Analysis Agent")
        if progress_callback:
            progress_callback("transaction", "running")

        t1 = time.time()
        results["transaction_analysis"] = self.transaction_agent.analyze(case_data)
        d1 = int((time.time() - t1) * 1000)
        self._log("Transaction Analysis Agent", "complete",
                  f"Risk score: {results['transaction_analysis']['risk_score']}/100", d1)

        if progress_callback:
            progress_callback("transaction", "complete")

        # Step 2: Customer Behaviour Analysis
        self._log("Orchestrator", "routing", "Routing to Customer Behaviour Agent")
        if progress_callback:
            progress_callback("customer", "running")

        t2 = time.time()
        results["customer_analysis"] = self.customer_agent.analyze(case_data)
        d2 = int((time.time() - t2) * 1000)
        self._log("Customer Behaviour Agent", "complete",
                  f"Anomaly level: {results['customer_analysis']['anomaly_level']}", d2)

        if progress_callback:
            progress_callback("customer", "complete")

        # Step 3: Conditional Routing - check if early escalation needed
        combined_risk = (
            results["transaction_analysis"]["risk_score"] * 0.4 +
            results["customer_analysis"]["trust_score_delta"] * 0.3
        )

        if combined_risk > 70:
            self._log("Orchestrator", "escalation_check",
                      "High combined risk detected - flagging for priority review")

        # Step 4: Risk & Policy Assessment
        self._log("Orchestrator", "routing", "Routing to Risk & Policy Agent")
        if progress_callback:
            progress_callback("risk", "running")

        t3 = time.time()
        results["risk_assessment"] = self.risk_agent.analyze(case_data, results)
        d3 = int((time.time() - t3) * 1000)
        self._log("Risk & Policy Agent", "complete",
                  f"Policy action: {results['risk_assessment']['policy_action']}", d3)

        if progress_callback:
            progress_callback("risk", "complete")

        # Step 5: Recommendation Generation
        self._log("Orchestrator", "routing", "Routing to Recommendation Agent")
        if progress_callback:
            progress_callback("recommendation", "running")

        t4 = time.time()
        results["recommendation"] = self.recommendation_agent.analyze(case_data, results)
        d4 = int((time.time() - t4) * 1000)
        self._log("Recommendation Agent", "complete",
                  f"Decision: {results['recommendation']['decision']}", d4)

        if progress_callback:
            progress_callback("recommendation", "complete")

        # Final summary
        total_time = int((time.time() - start_time) * 1000)
        self._log("Orchestrator", "complete",
                  f"Pipeline complete in {total_time}ms. Decision: {results['recommendation']['decision']}",
                  total_time)

        results["pipeline_log"] = self.pipeline_log
        results["total_duration_ms"] = total_time
        results["model_used"] = self.model_name

        return results
