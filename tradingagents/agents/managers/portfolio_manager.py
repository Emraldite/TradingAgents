"""Portfolio Manager: synthesises specialist evidence into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        reports = {
            "SEC Form 4 insider activity": state.get("insider_report", ""),
            "Market and technical evidence": state.get("market_report", ""),
            "Social sentiment": state.get("sentiment_report", ""),
            "Company and macro news": state.get("news_report", ""),
            "Fundamentals": state.get("fundamentals_report", ""),
        }
        evidence = "\n\n".join(
            f"## {name}\n{report or 'Unavailable; treat as neutral.'}"
            for name, report in reports.items()
        )

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the independent analyst reports and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

Compare the strongest bullish evidence with the strongest bearish evidence. Explicitly account for data quality, downside risk, conflicting signals, and missing evidence. Do not average to Hold automatically; use Hold only when the evidence is genuinely balanced. SEC data with no qualifying activity is neutral.

{lessons_line}
**Current evidence:**
{evidence}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        return {"final_trade_decision": final_trade_decision}

    return portfolio_manager_node
