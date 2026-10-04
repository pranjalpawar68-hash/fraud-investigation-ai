"""
Recommendation Agent
Synthesizes all agent outputs into a final recommendation for the human investigator.
"""

import time
import random


class RecommendationAgent:
    """Generates final recommendation with confidence and reasoning."""

    def __init__(self, model_name="Qwen-2.5-72B"):
        self.model_name = model_name
        self._model_params = {
            "Qwen-2.5-72B": {"precision": 0.93, "latency_base": 200},
            "Llama-3.1-70B": {"precision": 0.86, "latency_base": 240},
            "Mistral-Large-2": {"precision": 0.90, "latency_base": 185},
        }

    def _get_params(self):
        return self._model_params.get(self.model_name, self._model_params["Qwen-2.5-72B"])

    def analyze(self, case_data, prior_results):
        params = self._get_params()

        time.sleep(params["latency_base"] / 1000 + random.uniform(0.05, 0.15))

        txn_result = prior_results["transaction_analysis"]
        cust_result = prior_results["customer_analysis"]
        risk_result = prior_results["risk_assessment"]

        # --- Synthesize recommendation ---
        txn_risk = txn_result["risk_score"]
        cust_anomaly = cust_result["anomaly_score"]
        policy_action = risk_result["policy_action"]
        composite = risk_result["composite_risk_score"]

        # Decision logic
        if policy_action == "BLOCK_AND_ESCALATE":
            decision = "BLOCK & ESCALATE"
            confidence = min(98, 75 + int(composite * 0.25))
            urgency = "CRITICAL"
            actions = [
                "Immediately block the transaction",
                "Freeze account for 24 hours pending investigation",
                "Escalate to Senior Fraud Analyst (Tier 3)",
                "Initiate customer contact via registered phone number",
                "File Suspicious Transaction Report (STR) with FIU-IND",
                "Preserve all session/device forensic data",
            ]
        elif policy_action == "HOLD_AND_VERIFY":
            decision = "HOLD & VERIFY"
            confidence = min(90, 60 + int(composite * 0.2))
            urgency = "HIGH"
            actions = [
                "Place transaction on hold (4-hour SLA)",
                "Send OTP verification to registered mobile",
                "Request callback confirmation from customer",
                "Cross-verify beneficiary details",
                "If verified, release with monitoring flag; if unverified, escalate",
            ]
        elif policy_action == "FLAG_AND_MONITOR":
            decision = "APPROVE WITH FLAG"
            confidence = min(85, 55 + int((100 - composite) * 0.2))
            urgency = "MEDIUM"
            actions = [
                "Allow transaction to proceed",
                "Add 30-day enhanced monitoring on account",
                "Flag beneficiary for watchlist check",
                "Log case for periodic review",
            ]
        else:
            decision = "APPROVE"
            confidence = min(95, 70 + int((100 - composite) * 0.3))
            urgency = "LOW"
            actions = [
                "Approve transaction - within normal parameters",
                "No additional monitoring required",
                "Close alert with disposition: Legitimate Activity",
            ]

        # Evidence summary
        evidence_summary = {
            "transaction_risk_score": txn_risk,
            "customer_anomaly_score": cust_anomaly,
            "composite_risk_score": composite,
            "policy_violations_count": len(risk_result["policy_violations"]),
            "regulatory_flags_count": len(risk_result["regulatory_flags"]),
            "rule_triggers_count": len(case_data["rule_triggers"]),
            "key_risk_factors": (
                txn_result["findings"][:2] +
                cust_result["findings"][:2] +
                risk_result["findings"][:1]
            ),
        }

        # Hallucination check - flag any claims not supported by evidence
        hallucination_flags = []
        if txn_risk < 20 and decision in ["BLOCK & ESCALATE", "HOLD & VERIFY"]:
            hallucination_flags.append("Decision severity may exceed evidence from transaction analysis")
        if cust_anomaly < 15 and "account takeover" in str(cust_result.get("findings", "")).lower():
            hallucination_flags.append("Account takeover claim not strongly supported by customer signals")

        return {
            "agent": "Recommendation",
            "model": self.model_name,
            "decision": decision,
            "confidence": confidence,
            "urgency": urgency,
            "recommended_actions": actions,
            "evidence_summary": evidence_summary,
            "hallucination_flags": hallucination_flags,
            "hitl_required": risk_result["hitl_required"],
            "sla_hours": risk_result["sla_hours"],
            "structured_output_compliant": True,
            "explanation": self._generate_explanation(
                case_data, decision, confidence, urgency, actions, evidence_summary
            ),
        }

    def _generate_explanation(self, case, decision, confidence, urgency, actions, evidence):
        txn = case["transaction"]
        cust = case["customer"]

        explanations = {
            "Qwen-2.5-72B": (
                f"## Final Recommendation\n\n"
                f"**Case:** {case['case_id']} | **Customer:** {cust['name']} | "
                f"**Transaction:** {txn['txn_id']} (INR {txn['amount']:,})\n\n"
                f"**Decision: {decision}** (Confidence: {confidence}% | Urgency: {urgency})\n\n"
                f"**Rationale:** Based on comprehensive analysis across transaction patterns "
                f"(risk: {evidence['transaction_risk_score']}/100), customer behaviour "
                f"(anomaly: {evidence['customer_anomaly_score']}/100), and {evidence['policy_violations_count']} "
                f"policy violation(s), the system recommends **{decision}**. "
                f"The composite risk score of {evidence['composite_risk_score']}/100 "
                f"{'exceeds the automated approval threshold, mandating human review.' if evidence['composite_risk_score'] > 40 else 'is within automated approval parameters.'}\n\n"
                f"**Recommended Actions:**\n" + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions))
            ),
            "Llama-3.1-70B": (
                f"RECOMMENDATION REPORT\n"
                f"Case: {case['case_id']} | Txn: {txn['txn_id']} | Amount: INR {txn['amount']:,}\n"
                f"Decision: {decision} | Confidence: {confidence}% | Priority: {urgency}\n"
                f"Risk Scores - Transaction: {evidence['transaction_risk_score']}, "
                f"Customer: {evidence['customer_anomaly_score']}, "
                f"Composite: {evidence['composite_risk_score']}\n"
                f"Policy Violations: {evidence['policy_violations_count']} | "
                f"Regulatory Flags: {evidence['regulatory_flags_count']}\n"
                f"Actions: {'; '.join(actions[:3])}"
            ),
            "Mistral-Large-2": (
                f"[FRAUD INVESTIGATION RECOMMENDATION]\n"
                f"Case ID: {case['case_id']} | Subject: {cust['name']} ({cust['id']})\n"
                f"Transaction: {txn['txn_id']} - INR {txn['amount']:,} via {txn['type']}\n\n"
                f"VERDICT: {decision} (Confidence: {confidence}%, Urgency: {urgency})\n\n"
                f"Evidence Matrix: TXN Risk={evidence['transaction_risk_score']}/100 | "
                f"CUST Anomaly={evidence['customer_anomaly_score']}/100 | "
                f"Composite={evidence['composite_risk_score']}/100\n"
                f"Violations: {evidence['policy_violations_count']} | "
                f"Reg Flags: {evidence['regulatory_flags_count']}\n\n"
                f"Prescribed Actions:\n" + "\n".join(f"  - {a}" for a in actions)
            ),
        }
        return explanations.get(self.model_name, explanations["Qwen-2.5-72B"])
