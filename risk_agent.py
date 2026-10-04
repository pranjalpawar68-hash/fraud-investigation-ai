"""
Risk & Policy Agent
Evaluates case against bank policies, regulatory requirements, and risk thresholds.
"""

import time
import random


class RiskPolicyAgent:
    """Applies risk policies and regulatory rules to case evidence."""

    def __init__(self, model_name="Qwen-2.5-72B"):
        self.model_name = model_name
        self._model_params = {
            "Qwen-2.5-72B": {"precision": 0.94, "latency_base": 140},
            "Llama-3.1-70B": {"precision": 0.87, "latency_base": 175},
            "Mistral-Large-2": {"precision": 0.91, "latency_base": 135},
        }

    def _get_params(self):
        return self._model_params.get(self.model_name, self._model_params["Qwen-2.5-72B"])

    def analyze(self, case_data, prior_results):
        params = self._get_params()
        txn = case_data["transaction"]
        cust = case_data["customer"]
        rules = case_data["rule_triggers"]
        txn_risk = prior_results["transaction_analysis"]["risk_score"]
        cust_anomaly = prior_results["customer_analysis"]["anomaly_score"]

        time.sleep(params["latency_base"] / 1000 + random.uniform(0.05, 0.15))

        findings = []
        policy_violations = []
        regulatory_flags = []

        # --- Policy evaluation ---

        # 1. Transaction threshold policies
        if txn["amount"] > 200000:
            policy_violations.append("POL-001: Transaction exceeds INR 2,00,000 - mandatory enhanced due diligence")
            regulatory_flags.append("RBI AML/KYC Guidelines - Large value transaction reporting")
        if txn["amount"] > 1000000:
            policy_violations.append("POL-002: Transaction exceeds INR 10,00,000 - CTR (Cash Transaction Report) required")

        # 2. Suspicious pattern policies
        if txn_risk > 60 and cust_anomaly > 40:
            policy_violations.append("POL-010: Combined high transaction risk + customer anomaly - mandatory HITL review")
            findings.append("Combined risk indicators exceed automated approval threshold")

        # 3. Crypto/high-risk destination
        if "crypto" in txn.get("recipient", "").lower():
            policy_violations.append("POL-015: Transfer to cryptocurrency platform - enhanced monitoring required")
            regulatory_flags.append("FATF Travel Rule compliance check needed")

        # 4. Cross-border / VPN
        if "VPN" in txn.get("location", "") or "Nigeria" in txn.get("location", ""):
            policy_violations.append("POL-020: Foreign origin / VPN detected - cross-border fraud protocol")
            regulatory_flags.append("PMLA Section 12 - Suspicious Transaction Report required")
            findings.append("Transaction origin inconsistent with customer's registered jurisdiction")

        # 5. Elderly customer protection
        if cust["age"] > 60 and txn_risk > 40:
            policy_violations.append("POL-025: Elderly customer + elevated risk - mandatory telephonic verification")
            findings.append("Vulnerable customer protocol activated")

        # 6. New beneficiary + high value
        if txn["is_first_time_recipient"] and txn["amount"] > 50000:
            policy_violations.append("POL-030: High-value transfer to new beneficiary - cooling period advisory")
            findings.append("First-time beneficiary with significant amount warrants hold period")

        # Risk composite score
        composite_score = int(txn_risk * 0.45 + cust_anomaly * 0.35 + len(policy_violations) * 5)
        composite_score = min(composite_score, 100)

        # Determine policy action
        if composite_score >= 75 or any("mandatory" in v.lower() and "HITL" in v for v in policy_violations):
            policy_action = "BLOCK_AND_ESCALATE"
            sla_hours = 1
        elif composite_score >= 50:
            policy_action = "HOLD_AND_VERIFY"
            sla_hours = 4
        elif composite_score >= 25:
            policy_action = "FLAG_AND_MONITOR"
            sla_hours = 24
        else:
            policy_action = "APPROVE_WITH_NOTE"
            sla_hours = 0

        hitl_required = composite_score >= 40 or len(policy_violations) >= 2

        return {
            "agent": "Risk & Policy",
            "model": self.model_name,
            "composite_risk_score": composite_score,
            "policy_action": policy_action,
            "policy_violations": policy_violations,
            "regulatory_flags": regulatory_flags,
            "hitl_required": hitl_required,
            "sla_hours": sla_hours,
            "findings": findings,
            "rules_evaluated": len(rules),
            "policies_checked": 6,
            "structured_output_compliant": True,
            "explanation": self._generate_explanation(
                composite_score, policy_action, policy_violations, regulatory_flags, hitl_required
            ),
        }

    def _generate_explanation(self, score, action, violations, reg_flags, hitl):
        explanations = {
            "Qwen-2.5-72B": (
                f"Risk & Policy assessment yields composite score of **{score}/100**. "
                f"Recommended action: **{action}**. "
                f"{len(violations)} policy violation(s) identified. "
                f"{len(reg_flags)} regulatory flag(s) raised. "
                f"Human-in-the-loop review: {'REQUIRED' if hitl else 'Optional'}. "
                f"{'Immediate escalation to senior fraud analyst recommended due to policy mandate.' if action == 'BLOCK_AND_ESCALATE' else ''}"
            ),
            "Llama-3.1-70B": (
                f"Policy Engine Result: Score {score}/100 | Action: {action} | "
                f"Violations: {len(violations)} | Regulatory Flags: {len(reg_flags)} | "
                f"HITL: {'Yes' if hitl else 'No'}. "
                f"{'Urgent: Multiple policy thresholds breached.' if score > 60 else 'Within standard escalation parameters.'}"
            ),
            "Mistral-Large-2": (
                f"[Risk Assessment Complete] Composite: {score}/100 | Policy Action: {action} | "
                f"Violations: {', '.join(v.split(':')[0] for v in violations) if violations else 'None'} | "
                f"HITL Required: {'Yes' if hitl else 'No'} | "
                f"Regulatory: {len(reg_flags)} flag(s). "
                f"{'SLA: 1 hour - critical priority.' if action == 'BLOCK_AND_ESCALATE' else ''}"
            ),
        }
        return explanations.get(self.model_name, explanations["Qwen-2.5-72B"])
