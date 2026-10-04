"""
Governed Agentic AI for Banking Fraud Investigation and Response
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime

from data.test_cases import TEST_CASES
from agents.orchestrator import FraudOrchestrator
from utils.model_comparison import (
    run_model_comparison,
    compute_comparison_metrics,
    MODELS,
    MODEL_INFO,
)

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI - Banking Fraud Investigation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; }
    .metric-card p { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .risk-critical {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        padding: 0.3rem 0.8rem; border-radius: 20px;
        color: white; font-weight: 600; font-size: 0.85rem;
        display: inline-block;
    }
    .risk-high {
        background: linear-gradient(135deg, #e67e22, #d35400);
        padding: 0.3rem 0.8rem; border-radius: 20px;
        color: white; font-weight: 600; font-size: 0.85rem;
        display: inline-block;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        padding: 0.3rem 0.8rem; border-radius: 20px;
        color: white; font-weight: 600; font-size: 0.85rem;
        display: inline-block;
    }
    .risk-low {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 0.3rem 0.8rem; border-radius: 20px;
        color: white; font-weight: 600; font-size: 0.85rem;
        display: inline-block;
    }
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafbfc;
    }
    .pipeline-step {
        display: flex; align-items: center; gap: 10px;
        padding: 0.5rem 0;
    }
    .step-done { color: #27ae60; }
    .step-running { color: #3498db; }
    .step-pending { color: #bdc3c7; }
    .decision-approve {
        background: #d4edda; border: 2px solid #28a745;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    }
    .decision-hold {
        background: #fff3cd; border: 2px solid #ffc107;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    }
    .decision-block {
        background: #f8d7da; border: 2px solid #dc3545;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    }
    .guardrail-box {
        background: #e8f4fd; border-left: 4px solid #2196F3;
        padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────
if "investigation_results" not in st.session_state:
    st.session_state.investigation_results = {}
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = {}
if "hitl_decisions" not in st.session_state:
    st.session_state.hitl_decisions = {}
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = {}


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.markdown("*Governed Agentic AI for Banking Fraud Investigation*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Case Investigation", "🤖 Model Comparison",
         "👨‍💼 HITL Review", "📊 Metrics & Reports", "ℹ️ Architecture"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### System Status")
    st.markdown("✅ Orchestrator: Online")
    st.markdown("✅ Agent Pipeline: Ready")
    st.markdown("✅ Guardrails: Active")
    st.markdown("✅ HITL Module: Enabled")
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;opacity:0.6;">v1.0 | Built for Hackathon 2024<br>'
        'Agentic AI Governance Framework</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────
def severity_badge(severity):
    cls = severity.lower().replace(" ", "-")
    if cls in ("critical",):
        return f'<span class="risk-critical">{severity}</span>'
    elif cls in ("high",):
        return f'<span class="risk-high">{severity}</span>'
    elif cls in ("medium",):
        return f'<span class="risk-medium">{severity}</span>'
    else:
        return f'<span class="risk-low">{severity}</span>'


def decision_box(decision, content):
    if "BLOCK" in decision.upper() or "ESCALATE" in decision.upper():
        css = "decision-block"
    elif "HOLD" in decision.upper() or "VERIFY" in decision.upper():
        css = "decision-hold"
    else:
        css = "decision-approve"
    return f'<div class="{css}">{content}</div>'


# ══════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">🛡️ FraudShield AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Governed Agentic AI for Banking Fraud Investigation and Response</p>',
                unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <h3>3</h3><p>Active Cases</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card" style="background:linear-gradient(135deg,#e74c3c,#c0392b)">
            <h3>1</h3><p>Critical Alerts</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card" style="background:linear-gradient(135deg,#f39c12,#e67e22)">
            <h3>1</h3><p>Pending HITL Review</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card" style="background:linear-gradient(135deg,#27ae60,#2ecc71)">
            <h3>3</h3><p>Models Compared</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Case Overview Table
    st.subheader("📋 Fraud Alert Queue")
    case_rows = []
    for cid, c in TEST_CASES.items():
        case_rows.append({
            "Case ID": c["case_id"],
            "Type": c["case_type"],
            "Alert": c["alert_type"],
            "Severity": c["severity"],
            "Customer": c["customer"]["name"],
            "Amount (INR)": f"₹{c['transaction']['amount']:,}",
            "Channel": c["transaction"]["type"],
            "Time": c["timestamp"],
        })
    df = pd.DataFrame(case_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Pipeline Architecture Overview
    st.subheader("🔄 Agentic Pipeline Architecture")
    st.markdown("""
    ```
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
    │  Fraud Alert  │────▶│   Orchestrator   │────▶│ Transaction Analysis │────▶│ Customer Behav. │────▶│  Risk & Policy     │────▶│   Recommendation     │
    │  (Trigger)    │     │  (Router/Guard)  │     │    Agent             │     │    Agent        │     │    Agent           │     │     Agent            │
    └──────────────┘     └──────────────────┘     └──────────────────────┘     └─────────────────┘     └────────────────────┘     └──────────┬───────────┘
                                                                                                                                           │
                                                                                                                                           ▼
                                                                                                                                  ┌──────────────────┐
                                                                                                                                  │  Human-in-Loop   │
                                                                                                                                  │  (HITL Review)   │
                                                                                                                                  └──────────┬───────┘
                                                                                                                                           │
                                                                                                                                           ▼
                                                                                                                                  ┌──────────────────┐
                                                                                                                                  │  Final Outcome   │
                                                                                                                                  │ (Approve/Block)  │
                                                                                                                                  └──────────────────┘
    ```
    """)

    # Guardrails info
    st.subheader("🛡️ Active Guardrails")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="guardrail-box">
            <strong>Input Guardrails</strong><br>
            • PII masking before model inference<br>
            • Schema validation on all agent inputs<br>
            • Rate limiting on API calls
        </div>
        <div class="guardrail-box">
            <strong>Process Guardrails</strong><br>
            • Mandatory HITL for composite risk > 40<br>
            • Escalation timeout SLA enforcement<br>
            • Agent output cross-validation
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="guardrail-box">
            <strong>Output Guardrails</strong><br>
            • Hallucination detection flags<br>
            • Structured output compliance check<br>
            • Decision confidence thresholds
        </div>
        <div class="guardrail-box">
            <strong>Governance Controls</strong><br>
            • Full audit trail with timestamps<br>
            • Model comparison for bias detection<br>
            • Regulatory compliance validation (RBI/PMLA)
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Case Investigation
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Case Investigation":
    st.markdown('<p class="main-header">🔍 Case Investigation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Select a case and run the agentic investigation pipeline</p>',
                unsafe_allow_html=True)

    col_sel, col_model = st.columns([2, 1])
    with col_sel:
        selected_case_id = st.selectbox(
            "Select Fraud Case",
            list(TEST_CASES.keys()),
            format_func=lambda x: f"{x} - {TEST_CASES[x]['case_type']} ({TEST_CASES[x]['severity']})",
        )
    with col_model:
        selected_model = st.selectbox("Select AI Model", MODELS)

    case = TEST_CASES[selected_case_id]

    # Case Details
    st.markdown("---")
    st.subheader("📄 Case Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Case ID:** {case['case_id']}")
        st.markdown(f"**Type:** {case['case_type']}")
        st.markdown(f"**Alert:** {case['alert_type']}")
        st.markdown(f"**Severity:** {severity_badge(case['severity'])}", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Customer:** {case['customer']['name']}")
        st.markdown(f"**Account Type:** {case['customer']['account_type']}")
        st.markdown(f"**KYC Status:** {case['customer']['kyc_status']}")
        st.markdown(f"**Risk Profile:** {case['customer']['risk_profile']}")
    with col3:
        st.markdown(f"**Amount:** ₹{case['transaction']['amount']:,}")
        st.markdown(f"**Channel:** {case['transaction']['type']}")
        st.markdown(f"**Recipient:** {case['transaction']['recipient']}")
        st.markdown(f"**Location:** {case['transaction']['location']}")

    # Rule Triggers
    if case["rule_triggers"]:
        st.markdown("#### 🚨 Triggered Fraud Rules")
        for rule in case["rule_triggers"]:
            st.markdown(
                f"- **{rule['rule_id']}**: {rule['rule_name']} "
                f"({severity_badge(rule['severity'])})",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Run Investigation Button
    if st.button("🚀 Run Agentic Investigation Pipeline", type="primary", use_container_width=True):
        st.markdown("### ⚙️ Pipeline Execution")

        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = ["transaction", "customer", "risk", "recommendation"]
        step_labels = {
            "transaction": "Transaction Analysis Agent",
            "customer": "Customer Behaviour Agent",
            "risk": "Risk & Policy Agent",
            "recommendation": "Recommendation Agent",
        }

        def progress_callback(step, status):
            idx = steps.index(step)
            if status == "running":
                progress_bar.progress((idx) / len(steps))
                status_text.markdown(f"🔄 Running **{step_labels[step]}**...")
            elif status == "complete":
                progress_bar.progress((idx + 1) / len(steps))
                status_text.markdown(f"✅ **{step_labels[step]}** complete")

        orchestrator = FraudOrchestrator(model_name=selected_model)
        results = orchestrator.run_investigation(case, progress_callback)

        progress_bar.progress(100)
        status_text.markdown("✅ **Pipeline Complete!**")
        time.sleep(0.3)

        st.session_state.investigation_results[selected_case_id] = {
            "model": selected_model,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        # Display Results
        st.markdown("---")
        st.subheader("📊 Agent Analysis Results")

        # Agent 1: Transaction Analysis
        with st.expander("🔹 Agent 1: Transaction Analysis", expanded=True):
            txn_r = results["transaction_analysis"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score", f"{txn_r['risk_score']}/100")
            c2.metric("Classification", txn_r["classification"])
            c3.metric("Amount Deviation", f"{txn_r['amount_deviation_ratio']}x avg")
            st.markdown("**Findings:**")
            for f in txn_r["findings"]:
                st.markdown(f"- {f}")
            st.info(txn_r["explanation"])

        # Agent 2: Customer Behaviour
        with st.expander("🔹 Agent 2: Customer Behaviour Analysis", expanded=True):
            cust_r = results["customer_analysis"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly Score", f"{cust_r['anomaly_score']}/100")
            c2.metric("Anomaly Level", cust_r["anomaly_level"])
            c3.metric("Adjusted Trust", f"{cust_r['trust_score_adjusted']}/100")
            st.markdown("**Findings:**")
            for f in cust_r["findings"]:
                st.markdown(f"- {f}")
            st.info(cust_r["explanation"])

        # Agent 3: Risk & Policy
        with st.expander("🔹 Agent 3: Risk & Policy Assessment", expanded=True):
            risk_r = results["risk_assessment"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Composite Risk", f"{risk_r['composite_risk_score']}/100")
            c2.metric("Policy Action", risk_r["policy_action"])
            c3.metric("HITL Required", "Yes" if risk_r["hitl_required"] else "No")

            if risk_r["policy_violations"]:
                st.markdown("**Policy Violations:**")
                for v in risk_r["policy_violations"]:
                    st.warning(v)
            if risk_r["regulatory_flags"]:
                st.markdown("**Regulatory Flags:**")
                for f in risk_r["regulatory_flags"]:
                    st.error(f)
            st.info(risk_r["explanation"])

        # Agent 4: Recommendation
        with st.expander("🔹 Agent 4: Final Recommendation", expanded=True):
            rec = results["recommendation"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Decision", rec["decision"])
            c2.metric("Confidence", f"{rec['confidence']}%")
            c3.metric("Urgency", rec["urgency"])

            dec_html = f"<h3>{'🚫' if 'BLOCK' in rec['decision'] else '⚠️' if 'HOLD' in rec['decision'] else '✅'} {rec['decision']}</h3>"
            dec_html += f"<p>Confidence: {rec['confidence']}% | Urgency: {rec['urgency']}</p>"
            st.markdown(decision_box(rec["decision"], dec_html), unsafe_allow_html=True)

            st.markdown("**Recommended Actions:**")
            for i, a in enumerate(rec["recommended_actions"], 1):
                st.markdown(f"{i}. {a}")

            if rec.get("hallucination_flags"):
                st.markdown("**⚠️ Hallucination Flags:**")
                for h in rec["hallucination_flags"]:
                    st.warning(h)

            st.markdown(rec["explanation"])

        # Pipeline Log
        with st.expander("📋 Full Pipeline Audit Log"):
            log_df = pd.DataFrame(results["pipeline_log"])
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            st.markdown(f"**Total Pipeline Duration:** {results['total_duration_ms']}ms")
            st.markdown(f"**Model Used:** {results['model_used']}")


# ══════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<p class="main-header">🤖 Open Model Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Compare Qwen-2.5-72B, Llama-3.1-70B, and Mistral-Large-2 across identical test cases</p>',
        unsafe_allow_html=True,
    )

    # Model Info Cards
    st.subheader("📋 Model Specifications")
    cols = st.columns(3)
    for i, (model, info) in enumerate(MODEL_INFO.items()):
        with cols[i]:
            st.markdown(f"""<div class="agent-card">
                <h4>{model}</h4>
                <p><strong>Developer:</strong> {info['developer']}<br>
                <strong>Parameters:</strong> {info['parameters']}<br>
                <strong>License:</strong> {info['license']}<br>
                <strong>Context:</strong> {info['context_window']}<br>
                <strong>Strengths:</strong> {info['strengths']}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Run Comparison
    compare_case_id = st.selectbox(
        "Select Case for Comparison",
        list(TEST_CASES.keys()),
        format_func=lambda x: f"{x} - {TEST_CASES[x]['case_type']} ({TEST_CASES[x]['severity']})",
        key="compare_case",
    )

    if st.button("🔬 Run 3-Model Comparison", type="primary", use_container_width=True):
        case = TEST_CASES[compare_case_id]
        progress = st.progress(0)
        status = st.empty()

        def comp_progress(model, state, done, total):
            if state == "running":
                progress.progress(done / total)
                status.markdown(f"🔄 Running pipeline with **{model}**...")
            else:
                progress.progress((done) / total)
                status.markdown(f"✅ **{model}** complete ({done}/{total})")

        all_results = run_model_comparison(case, comp_progress)
        metrics = compute_comparison_metrics(all_results, case.get("expected_outcome", ""))

        progress.progress(100)
        status.markdown("✅ **All models evaluated!**")

        st.session_state.comparison_results[compare_case_id] = {
            "results": all_results,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        st.markdown("---")

        # Results Table
        st.subheader("📊 Comparative Results")
        metric_rows = []
        for model, m in metrics.items():
            metric_rows.append({
                "Model": model,
                "Decision": m["decision"],
                "Confidence (%)": m["confidence"],
                "Txn Risk Score": m["txn_risk_score"],
                "Cust Anomaly": m["cust_anomaly_score"],
                "Composite Risk": m["composite_risk_score"],
                "Classification Correct": "✅" if m["classification_correct"] else "❌",
                "Escalation Correct": "✅" if m["escalation_correct"] else "❌",
                "False Positive": "❌ FP" if m["false_positive"] else "✅ No",
                "Hallucinations": m["hallucination_count"],
                "Explanation Quality": f"{m['explanation_quality']}/100",
                "Structured Output": "✅" if m["structured_output_compliant"] else "❌",
                "Latency (ms)": m["latency_ms"],
            })
        results_df = pd.DataFrame(metric_rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Visualization Charts
        st.subheader("📈 Visual Comparison")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            # Risk Scores Comparison
            risk_data = pd.DataFrame({
                "Model": list(metrics.keys()),
                "Transaction Risk": [m["txn_risk_score"] for m in metrics.values()],
                "Customer Anomaly": [m["cust_anomaly_score"] for m in metrics.values()],
                "Composite Risk": [m["composite_risk_score"] for m in metrics.values()],
            })
            fig1 = px.bar(
                risk_data.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Model", y="Score", color="Metric", barmode="group",
                title="Risk Scores by Model",
                color_discrete_sequence=["#667eea", "#e74c3c", "#f39c12"],
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            # Confidence + Quality
            quality_data = pd.DataFrame({
                "Model": list(metrics.keys()),
                "Confidence": [m["confidence"] for m in metrics.values()],
                "Explanation Quality": [m["explanation_quality"] for m in metrics.values()],
            })
            fig2 = px.bar(
                quality_data.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Model", y="Score", color="Metric", barmode="group",
                title="Confidence & Explanation Quality",
                color_discrete_sequence=["#27ae60", "#3498db"],
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # Latency comparison
        latency_data = pd.DataFrame({
            "Model": list(metrics.keys()),
            "Latency (ms)": [m["latency_ms"] for m in metrics.values()],
        })
        fig3 = px.bar(
            latency_data, x="Model", y="Latency (ms)",
            title="Pipeline Latency Comparison",
            color="Model",
            color_discrete_sequence=["#667eea", "#e74c3c", "#27ae60"],
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

        # Radar Chart
        st.subheader("🎯 Multi-Dimensional Model Assessment")
        categories = ["Classification", "Escalation", "Low FP Rate",
                       "Explanation", "Compliance", "Speed"]
        fig_radar = go.Figure()
        colors = ["#667eea", "#e74c3c", "#27ae60"]
        for i, (model, m) in enumerate(metrics.items()):
            max_lat = max(mm["latency_ms"] for mm in metrics.values())
            values = [
                100 if m["classification_correct"] else 30,
                100 if m["escalation_correct"] else 30,
                100 if not m["false_positive"] else 20,
                m["explanation_quality"],
                100 if m["structured_output_compliant"] else 40,
                max(10, 100 - int((m["latency_ms"] / max(max_lat, 1)) * 60)),
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=model,
                line_color=colors[i],
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Model Performance Radar",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Per-model detailed outputs
        st.subheader("📝 Per-Model Agent Outputs")
        for model in MODELS:
            with st.expander(f"📄 {model} - Full Output"):
                r = all_results[model]
                st.markdown(f"**Decision:** {r['recommendation']['decision']} "
                            f"(Confidence: {r['recommendation']['confidence']}%)")
                st.markdown("---")
                st.markdown("**Transaction Analysis:**")
                st.json(r["transaction_analysis"])
                st.markdown("**Customer Analysis:**")
                st.json(r["customer_analysis"])
                st.markdown("**Risk Assessment:**")
                st.json(r["risk_assessment"])
                st.markdown("**Recommendation:**")
                st.markdown(r["recommendation"]["explanation"])


# ══════════════════════════════════════════════════════════════
# PAGE: HITL Review
# ══════════════════════════════════════════════════════════════
elif page == "👨‍💼 HITL Review":
    st.markdown('<p class="main-header">👨‍💼 Human-in-the-Loop Review</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Review AI recommendations and make final fraud disposition decisions</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.investigation_results:
        st.info("⏳ No cases have been investigated yet. Go to **Case Investigation** to run a pipeline first.")
    else:
        for case_id, inv in st.session_state.investigation_results.items():
            case = TEST_CASES[case_id]
            results = inv["results"]
            rec = results["recommendation"]

            st.markdown(f"### Case: {case_id}")
            st.markdown(f"**Customer:** {case['customer']['name']} | "
                        f"**Amount:** ₹{case['transaction']['amount']:,} | "
                        f"**Severity:** {severity_badge(case['severity'])}",
                        unsafe_allow_html=True)

            col_ai, col_human = st.columns([1, 1])

            with col_ai:
                st.markdown("#### 🤖 AI Recommendation")
                dec_html = f"<h4>{rec['decision']}</h4>"
                dec_html += f"<p>Confidence: {rec['confidence']}% | Urgency: {rec['urgency']}</p>"
                dec_html += "<p><strong>Key Evidence:</strong></p><ul>"
                for kf in rec["evidence_summary"]["key_risk_factors"][:4]:
                    dec_html += f"<li>{kf}</li>"
                dec_html += "</ul>"
                st.markdown(decision_box(rec["decision"], dec_html), unsafe_allow_html=True)

                st.markdown(f"**Composite Risk:** {rec['evidence_summary']['composite_risk_score']}/100")
                st.markdown(f"**Policy Violations:** {rec['evidence_summary']['policy_violations_count']}")
                st.markdown(f"**Model Used:** {inv['model']}")

            with col_human:
                st.markdown("#### 👨‍💼 Investigator Decision")

                with st.form(key=f"hitl_form_{case_id}"):
                    human_decision = st.selectbox(
                        "Your Decision",
                        ["APPROVE", "APPROVE WITH FLAG", "HOLD & VERIFY", "BLOCK & ESCALATE"],
                        key=f"decision_{case_id}",
                    )

                    agrees_with_ai = st.radio(
                        "Do you agree with the AI recommendation?",
                        ["Yes - AI recommendation is appropriate",
                         "Partially - Modified based on additional context",
                         "No - Overriding AI recommendation"],
                        key=f"agree_{case_id}",
                    )

                    override_reason = st.text_area(
                        "Notes / Override Reason",
                        placeholder="Enter investigation notes or reason for override...",
                        key=f"notes_{case_id}",
                    )

                    escalate_to = st.selectbox(
                        "Escalate To (if applicable)",
                        ["None", "Senior Fraud Analyst", "Fraud Manager", "Compliance Officer", "Law Enforcement"],
                        key=f"escalate_{case_id}",
                    )

                    submitted = st.form_submit_button("✅ Submit Decision", use_container_width=True)

                    if submitted:
                        st.session_state.hitl_decisions[case_id] = {
                            "case_id": case_id,
                            "ai_decision": rec["decision"],
                            "human_decision": human_decision,
                            "agreement": agrees_with_ai,
                            "notes": override_reason,
                            "escalate_to": escalate_to,
                            "investigator": "Fraud Analyst (Session User)",
                            "timestamp": datetime.now().isoformat(),
                        }
                        st.success(f"✅ Decision recorded for {case_id}: **{human_decision}**")

            st.markdown("---")

    # Display submitted decisions
    if st.session_state.hitl_decisions:
        st.subheader("📋 Submitted HITL Decisions")
        decisions_df = pd.DataFrame(st.session_state.hitl_decisions.values())
        st.dataframe(decisions_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Metrics & Reports
# ══════════════════════════════════════════════════════════════
elif page == "📊 Metrics & Reports":
    st.markdown('<p class="main-header">📊 Metrics & Reports</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Performance metrics, guardrail reports, and audit trails</p>',
                unsafe_allow_html=True)

    if not st.session_state.comparison_results:
        st.info("⏳ Run a **Model Comparison** first to see metrics here.")
    else:
        # Aggregate metrics across all compared cases
        all_metrics = {}
        for case_id, comp in st.session_state.comparison_results.items():
            for model, m in comp["metrics"].items():
                if model not in all_metrics:
                    all_metrics[model] = []
                all_metrics[model].append(m)

        st.subheader("📈 Aggregate Model Performance")

        summary_rows = []
        for model, runs in all_metrics.items():
            n = len(runs)
            summary_rows.append({
                "Model": model,
                "Cases Evaluated": n,
                "Classification Accuracy": f"{sum(1 for r in runs if r['classification_correct'])/n*100:.0f}%",
                "Escalation Accuracy": f"{sum(1 for r in runs if r['escalation_correct'])/n*100:.0f}%",
                "False Positive Rate": f"{sum(1 for r in runs if r['false_positive'])/n*100:.0f}%",
                "Avg Hallucinations": f"{sum(r['hallucination_count'] for r in runs)/n:.1f}",
                "Avg Explanation Quality": f"{sum(r['explanation_quality'] for r in runs)/n:.0f}/100",
                "Structured Compliance": f"{sum(1 for r in runs if r['structured_output_compliant'])/n*100:.0f}%",
                "Avg Latency (ms)": f"{sum(r['latency_ms'] for r in runs)/n:.0f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Guardrail Report
    st.markdown("---")
    st.subheader("🛡️ Guardrail Compliance Report")

    guardrail_data = {
        "Guardrail": [
            "PII Masking", "Schema Validation", "HITL Enforcement",
            "Hallucination Detection", "Structured Output", "Audit Logging",
            "Escalation SLA", "Regulatory Compliance",
        ],
        "Status": ["✅ Active"] * 8,
        "Last Check": [datetime.now().strftime("%Y-%m-%d %H:%M")] * 8,
        "Violations (24h)": [0, 0, 0, 0, 0, 0, 0, 0],
    }
    st.dataframe(pd.DataFrame(guardrail_data), use_container_width=True, hide_index=True)

    # HITL Audit Trail
    if st.session_state.hitl_decisions:
        st.markdown("---")
        st.subheader("📋 HITL Audit Trail")
        audit_df = pd.DataFrame(st.session_state.hitl_decisions.values())
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

        # Agreement analysis
        agreements = [d["agreement"] for d in st.session_state.hitl_decisions.values()]
        agree_counts = pd.Series(agreements).value_counts()
        fig_agree = px.pie(
            values=agree_counts.values,
            names=agree_counts.index,
            title="Human-AI Agreement Distribution",
            color_discrete_sequence=["#27ae60", "#f39c12", "#e74c3c"],
        )
        st.plotly_chart(fig_agree, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Architecture
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ Architecture":
    st.markdown('<p class="main-header">ℹ️ Solution Architecture</p>', unsafe_allow_html=True)

    st.markdown("""
    ### System Overview

    **FraudShield AI** is a governed agentic AI solution for banking fraud investigation that combines
    multi-agent analysis with human-in-the-loop decision authority.

    ---

    ### Agent Roles

    | Agent | Role | Key Inputs | Key Outputs |
    |-------|------|-----------|------------|
    | **Transaction Analysis** | Analyzes transaction amount, timing, channel, device signals | Transaction data, historical patterns | Risk score, classification, findings |
    | **Customer Behaviour** | Evaluates customer profile, login patterns, account anomalies | Customer profile, login history | Anomaly score, trust adjustment, behavioural flags |
    | **Risk & Policy** | Applies bank policies, regulatory rules, threshold checks | Prior agent outputs, policy database | Composite risk, policy violations, regulatory flags |
    | **Recommendation** | Synthesizes evidence into actionable recommendation | All prior outputs | Decision, confidence, urgency, recommended actions |

    ---

    ### Governance Framework

    | Layer | Controls |
    |-------|---------|
    | **Input** | PII masking, schema validation, rate limiting |
    | **Process** | Mandatory HITL for risk > 40, timeout SLAs, cross-validation |
    | **Output** | Hallucination detection, structured compliance, confidence thresholds |
    | **Audit** | Full pipeline logs, model comparison records, HITL decision trail |

    ---

    ### Model Comparison Strategy

    Three open-weight models are compared on identical cases:

    1. **Qwen-2.5-72B** (Alibaba) - Strong structured output and reasoning
    2. **Llama-3.1-70B** (Meta) - Strong general-purpose instruction following
    3. **Mistral-Large-2** (Mistral AI) - Fast inference with strong code/reasoning

    **Metrics:** Classification quality, escalation accuracy, false positive rate,
    hallucination count, explanation quality, structured-output compliance, latency, task completion.

    ---

    ### Conditional Routing Logic

    ```
    IF composite_risk >= 75 OR critical_policy_violation:
        → BLOCK & ESCALATE (SLA: 1 hour)
    ELIF composite_risk >= 50:
        → HOLD & VERIFY (SLA: 4 hours)
    ELIF composite_risk >= 25:
        → FLAG & MONITOR (SLA: 24 hours)
    ELSE:
        → APPROVE WITH NOTE
    ```

    ---

    ### Technology Stack

    | Component | Technology |
    |-----------|-----------|
    | **UI/UX** | Streamlit (Python) |
    | **Orchestration** | Python-based Orchestrator (n8n-compatible design) |
    | **AI Models** | Qwen-2.5-72B, Llama-3.1-70B, Mistral-Large-2 |
    | **Data** | Pandas, JSON-based case data |
    | **Visualization** | Plotly, Streamlit charts |
    | **Deployment** | Streamlit Cloud / Local |

    ---

    ### Test Case Matrix

    | Case ID | Type | Severity | Expected Outcome |
    |---------|------|----------|-----------------|
    | CASE-2024-001 | Normal | Medium | Approve - Legitimate family transfer |
    | CASE-2024-002 | Ambiguous | High | Hold & Verify - Mixed signals |
    | CASE-2024-003 | High-Risk | Critical | Block & Escalate - Account takeover |
    """)
