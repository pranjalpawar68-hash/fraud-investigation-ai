"""
Customer Behaviour Agent
Analyzes customer profile, history, and behavioural anomalies.
"""

import time
import random


class CustomerBehaviourAgent:
    """Evaluates customer behaviour patterns and trust indicators."""

    def __init__(self, model_name="Qwen-2.5-72B"):
        self.model_name = model_name
        self._model_params = {
            "Qwen-2.5-72B": {"precision": 0.93, "latency_base": 160},
            "Llama-3.1-70B": {"precision": 0.88, "latency_base": 195},
            "Mistral-Large-2": {"precision": 0.91, "latency_base": 150},
        }

    def _get_params(self):
        return self._model_params.get(self.model_name, self._model_params["Qwen-2.5-72B"])

    def analyze(self, case_data):
        params = self._get_params()
        cust = case_data["customer"]
        hist = case_data["historical_context"]
        txn = case_data["transaction"]

        time.sleep(params["latency_base"] / 1000 + random.uniform(0.05, 0.15))

        # --- Behavioural analysis ---
        anomaly_score = 0
        findings = []

        # 1. Account maturity trust
        if cust["account_age_years"] > 10:
            trust_base = 85
            findings.append(f"Long-standing customer ({cust['account_age_years']} years) - high baseline trust")
        elif cust["account_age_years"] > 5:
            trust_base = 70
            findings.append(f"Established customer ({cust['account_age_years']} years) - good baseline trust")
        elif cust["account_age_years"] > 1:
            trust_base = 50
            findings.append(f"Moderate tenure ({cust['account_age_years']} years) - standard baseline trust")
        else:
            trust_base = 30
            findings.append(f"New customer ({cust['account_age_years']} year(s)) - lower baseline trust")

        # 2. Login anomalies
        failed_logins = hist["failed_login_attempts_24h"]
        if failed_logins > 5:
            anomaly_score += 30
            findings.append(f"{failed_logins} failed login attempts in 24h - strong brute-force indicator")
        elif failed_logins > 2:
            anomaly_score += 15
            findings.append(f"{failed_logins} failed login attempts in 24h - potential credential testing")
        elif failed_logins > 0:
            anomaly_score += 5
            findings.append(f"{failed_logins} failed login attempt(s) in 24h - minimal concern")

        # 3. Password / device changes
        if hist["password_changed_recently"] and hist["device_change_recent"]:
            anomaly_score += 25
            findings.append("Both password AND device changed recently - suspicious credential takeover pattern")
        elif hist["password_changed_recently"]:
            anomaly_score += 10
            findings.append("Password changed recently - monitor for unauthorized access")
        elif hist["device_change_recent"]:
            anomaly_score += 10
            findings.append("New device detected - could indicate device compromise or legitimate upgrade")

        # 4. Prior fraud history
        if hist["previous_fraud_cases"] > 0:
            anomaly_score += 15
            findings.append(f"Customer has {hist['previous_fraud_cases']} prior fraud case(s) on record")
        else:
            findings.append("No prior fraud cases - clean history")

        # 5. Customer vulnerability assessment
        if cust["age"] > 60:
            anomaly_score += 10
            findings.append(f"Elderly customer (age {cust['age']}) - heightened vulnerability to social engineering/vishing")
        elif cust["age"] < 25:
            anomaly_score += 3
            findings.append(f"Young customer (age {cust['age']}) - consider impersonation risk")

        # 6. Behavioural consistency
        if hist["similar_txns_last_6mo"] == 0:
            anomaly_score += 15
            findings.append("No similar transactions in last 6 months - completely new pattern")
        elif hist["similar_txns_last_6mo"] < 3:
            anomaly_score += 5
            findings.append(f"Only {hist['similar_txns_last_6mo']} similar transactions in 6 months - infrequent pattern")
        else:
            findings.append(f"{hist['similar_txns_last_6mo']} similar transactions in 6 months - established pattern")

        anomaly_score = min(anomaly_score, 100)
        trust_delta = max(0, min(100, anomaly_score))

        # Anomaly classification
        if anomaly_score >= 60:
            anomaly_level = "Critical"
        elif anomaly_score >= 35:
            anomaly_level = "Elevated"
        elif anomaly_score >= 15:
            anomaly_level = "Moderate"
        else:
            anomaly_level = "Low"

        adjusted_trust = max(0, trust_base - int(anomaly_score * 0.6))

        return {
            "agent": "Customer Behaviour",
            "model": self.model_name,
            "trust_score_base": trust_base,
            "trust_score_adjusted": adjusted_trust,
            "trust_score_delta": trust_delta,
            "anomaly_score": anomaly_score,
            "anomaly_level": anomaly_level,
            "customer_risk_profile": cust["risk_profile"],
            "account_tenure_years": cust["account_age_years"],
            "findings": findings,
            "structured_output_compliant": True,
            "explanation": self._generate_explanation(cust, anomaly_level, anomaly_score, adjusted_trust, findings),
        }

    def _generate_explanation(self, cust, anomaly_level, score, trust, findings):
        explanations = {
            "Qwen-2.5-72B": (
                f"Customer {cust['name']} (ID: {cust['id']}) profile analysis reveals "
                f"**{anomaly_level}** anomaly level (score: {score}/100). "
                f"Adjusted trust score: {trust}/100 (base: {cust['risk_profile']} risk profile). "
                f"Key observations: {'; '.join(findings[:3])}. "
                f"{'Customer behaviour significantly deviates from established patterns - heightened scrutiny recommended.' if score > 40 else 'Behaviour broadly consistent with customer profile.'}"
            ),
            "Llama-3.1-70B": (
                f"Customer Profile Assessment: {cust['name']} | Account: {cust['account_type']} | "
                f"Tenure: {cust['account_age_years']}y | Anomaly: {anomaly_level} ({score}). "
                f"Trust adjustment: {trust}/100. "
                f"Findings: {'; '.join(findings[:3])}."
            ),
            "Mistral-Large-2": (
                f"[Customer Analysis] {cust['id']} - {cust['name']} | {cust['occupation']} | "
                f"Location: {cust['city']} | Anomaly Level: {anomaly_level} ({score}/100) | "
                f"Trust: {trust}/100. Key signals: {'; '.join(findings[:3])}. "
                f"{'Recommend verification contact.' if score > 30 else 'No immediate customer-side concerns.'}"
            ),
        }
        return explanations.get(self.model_name, explanations["Qwen-2.5-72B"])
