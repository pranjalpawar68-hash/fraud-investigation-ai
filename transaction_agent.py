"""
Transaction Analysis Agent
Analyzes transaction patterns, amounts, timing, and channel signals.
"""

import time
import random


class TransactionAnalysisAgent:
    """Evaluates transaction data for fraud indicators."""

    def __init__(self, model_name="Qwen-2.5-72B"):
        self.model_name = model_name
        # Model-specific variation factors for realistic comparison
        self._model_params = {
            "Qwen-2.5-72B": {"precision": 0.92, "latency_base": 180, "weight_amount": 0.35},
            "Llama-3.1-70B": {"precision": 0.89, "latency_base": 210, "weight_amount": 0.30},
            "Mistral-Large-2": {"precision": 0.90, "latency_base": 165, "weight_amount": 0.33},
        }

    def _get_params(self):
        return self._model_params.get(self.model_name, self._model_params["Qwen-2.5-72B"])

    def analyze(self, case_data):
        params = self._get_params()
        txn = case_data["transaction"]
        hist = case_data["historical_context"]
        rules = case_data["rule_triggers"]

        # Simulate model processing time
        time.sleep(params["latency_base"] / 1000 + random.uniform(0.05, 0.15))

        # --- Core risk scoring logic ---
        risk_score = 0
        findings = []

        # 1. Amount deviation
        amount_ratio = txn["amount"] / max(hist["avg_txn_amount"], 1)
        if amount_ratio > 10:
            risk_score += 35
            findings.append(f"Transaction amount is {amount_ratio:.1f}x the customer average - extreme deviation")
        elif amount_ratio > 3:
            risk_score += 20
            findings.append(f"Transaction amount is {amount_ratio:.1f}x the customer average - significant deviation")
        elif amount_ratio > 1.5:
            risk_score += 8
            findings.append(f"Transaction amount is {amount_ratio:.1f}x the customer average - mild deviation")
        else:
            findings.append(f"Transaction amount ({amount_ratio:.1f}x avg) is within normal range")

        # 2. Time-of-day analysis
        hour = int(txn["time_of_day"].split(":")[0])
        if "AM" in txn["time_of_day"] and hour < 6:
            risk_score += 15
            findings.append(f"Transaction at {txn['time_of_day']} - unusual late-night/early-morning activity")
        elif "AM" in txn["time_of_day"] and hour < 8:
            risk_score += 5
            findings.append(f"Transaction at {txn['time_of_day']} - early morning, slightly atypical")

        # 3. Channel and device signals
        if not txn["device_fingerprint_match"]:
            risk_score += 20
            findings.append("Device fingerprint does NOT match any known customer device")
        else:
            findings.append("Device fingerprint matches known customer device")

        if txn["is_first_time_recipient"]:
            risk_score += 10
            findings.append("First-time recipient - no prior transaction history with this beneficiary")

        # 4. Location analysis
        if "VPN" in txn.get("location", ""):
            risk_score += 25
            findings.append(f"VPN detected from {txn['location']} - IP origin masking")
        elif txn.get("location", "") and case_data["customer"]["city"] not in txn["location"]:
            risk_score += 10
            findings.append(f"Geo-location mismatch: customer city {case_data['customer']['city']} vs txn location {txn['location']}")

        # 5. Recipient risk
        recipient = txn.get("recipient", "").lower()
        if "crypto" in recipient:
            risk_score += 20
            findings.append("Recipient is a cryptocurrency exchange - high-risk destination category")
        elif "unregistered" in recipient.lower():
            risk_score += 5
            findings.append("Recipient is an unregistered beneficiary")

        # 6. Rule trigger count
        critical_rules = sum(1 for r in rules if r["severity"] == "Critical")
        high_rules = sum(1 for r in rules if r["severity"] == "High")
        if critical_rules > 0:
            risk_score += critical_rules * 8
            findings.append(f"{critical_rules} critical fraud rules triggered")
        if high_rules > 0:
            risk_score += high_rules * 4
            findings.append(f"{high_rules} high-severity fraud rules triggered")

        risk_score = min(risk_score, 100)

        # Model-specific adjustments
        noise = random.uniform(-3, 3) * (1 - params["precision"])
        risk_score = max(0, min(100, int(risk_score + noise)))

        # Classification
        if risk_score >= 75:
            classification = "High Risk"
        elif risk_score >= 45:
            classification = "Medium Risk"
        elif risk_score >= 20:
            classification = "Low Risk"
        else:
            classification = "Minimal Risk"

        return {
            "agent": "Transaction Analysis",
            "model": self.model_name,
            "risk_score": risk_score,
            "classification": classification,
            "amount_deviation_ratio": round(amount_ratio, 2),
            "findings": findings,
            "signals_detected": len(findings),
            "rule_triggers_analyzed": len(rules),
            "structured_output_compliant": True,
            "explanation": self._generate_explanation(case_data, risk_score, classification, findings),
        }

    def _generate_explanation(self, case, score, classification, findings):
        txn = case["transaction"]
        explanations = {
            "Qwen-2.5-72B": (
                f"Transaction {txn['txn_id']} for INR {txn['amount']:,} via {txn['type']} has been assessed "
                f"as **{classification}** (score: {score}/100). "
                f"Key factors: {'; '.join(findings[:3])}. "
                f"The transaction {'significantly deviates from' if score > 50 else 'is broadly consistent with'} "
                f"the customer's established behavioural baseline."
            ),
            "Llama-3.1-70B": (
                f"Analysis of TXN {txn['txn_id']}: Amount INR {txn['amount']:,} sent via {txn['type']}. "
                f"Risk Classification: {classification} ({score}/100). "
                f"Primary signals: {'; '.join(findings[:3])}. "
                f"Recommendation: {'Immediate review required' if score > 60 else 'Standard processing appropriate'}."
            ),
            "Mistral-Large-2": (
                f"[Transaction Risk Assessment] ID: {txn['txn_id']} | Amount: INR {txn['amount']:,} | "
                f"Channel: {txn['type']} | Score: {score}/100 ({classification}). "
                f"Detected {len(findings)} signals. Top findings: {'; '.join(findings[:3])}. "
                f"{'Escalation warranted based on cumulative risk indicators.' if score > 50 else 'Risk within acceptable thresholds.'}"
            ),
        }
        return explanations.get(self.model_name, explanations["Qwen-2.5-72B"])
