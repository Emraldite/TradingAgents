from typing import Optional
import datetime
import logging
import typer
import questionary
from pathlib import Path
from functools import wraps
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
    pretty_exceptions_show_locals=False,
)


def _configure_bot_logging(log_file: Path) -> Path:
    resolved = log_file.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        resolved,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return resolved


def _download_yfinance_quiet(yf, *args, **kwargs):
    """Suppress yfinance's duplicate stderr errors; callers report failures once."""
    yfinance_logger = logging.getLogger("yfinance")
    previous_level = yfinance_logger.level
    yfinance_logger.setLevel(logging.CRITICAL)
    try:
        return yf.download(*args, **kwargs)
    finally:
        yfinance_logger.setLevel(previous_level)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "insider": "Insider Analyst",
        "market": "Market Analyst",
        "social": "Sentiment Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "insider_report": ("insider", "Insider Analyst"),
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Sentiment Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Sentiment Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    from cli.announcements import fetch_announcements, display_announcements

    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", "r", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the exact ticker symbol to analyze, including exchange suffix when needed (examples: SPY, CNC.TO, 7203.T, 0700.HK)",
            "SPY",
        )
    )
    selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker)
    console.print(
        f"[green]Detected asset type:[/green] {asset_type.value}"
    )

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Output language
    console.print(
        create_question_box(
            "Step 3: Output Language",
            "Select the language for analyst reports and final decision"
        )
    )
    output_language = ask_output_language()

    # Step 4: Select analysts
    console.print(
        create_question_box(
            "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts(asset_type)
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: Research depth
    console.print(
        create_question_box(
            "Step 5: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 6: LLM Provider
    console.print(
        create_question_box(
            "Step 6: LLM Provider", "Select your LLM provider"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()

    # Providers with regional endpoints prompt for the region as a secondary
    # step so the main dropdown stays clean (mainland China and international
    # accounts cannot share API keys).
    if selected_llm_provider == "qwen":
        selected_llm_provider, backend_url = ask_qwen_region()
    elif selected_llm_provider == "minimax":
        selected_llm_provider, backend_url = ask_minimax_region()
    elif selected_llm_provider == "glm":
        selected_llm_provider, backend_url = ask_glm_region()

    # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
    # before model selection so it's obvious where we're connecting.
    if selected_llm_provider == "ollama":
        confirm_ollama_endpoint(backend_url)

    # Confirm the provider's API key is present; prompt the user to paste
    # one and persist it to .env if it's missing, so the analysis run
    # doesn't fail later at the first API call.
    ensure_api_key(selected_llm_provider)

    # Step 7: Thinking agents
    console.print(
        create_question_box(
            "Step 7: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 8: Provider-specific thinking configuration
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(
            create_question_box(
                "Step 8: Thinking Mode",
                "Configure Gemini thinking mode"
            )
        )
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box(
                "Step 8: Reasoning Effort",
                "Configure OpenAI reasoning effort level"
            )
        )
        reasoning_effort = ask_openai_reasoning_effort()
    elif provider_lower == "anthropic":
        console.print(
            create_question_box(
                "Step 8: Effort Level",
                "Configure Claude effort level"
            )
        )
        anthropic_effort = ask_anthropic_effort()

    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def get_ticker():
    """Get ticker symbol from user input, preserving exchange suffixes."""
    # typer.prompt strips trailing dot-suffixes on some shells (e.g. 000404.SH
    # collapses to 000404). questionary.text reads the raw line.
    ticker = questionary.text(
        "",
        validate=lambda value: (
            not value.strip()
            or (
                all(ch.isalnum() or ch in "._-^" for ch in value.strip())
                and len(value.strip()) <= 32
            )
        )
        or "Please enter a valid ticker symbol, e.g. AAPL, 000404.SZ, 0700.HK.",
    ).ask()

    if ticker is None:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        raise typer.Exit(1)

    return (ticker.strip() or "SPY").upper()


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save complete analysis report to disk with organized subfolders."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk, wall_time_tracker=None):
    """Update analyst statuses based on accumulated report state.

    Logic:
    - Store new report content from the current chunk if present
    - Check accumulated report_sections (not just current chunk) for status
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    if wall_time_tracker is not None:
        from tradingagents.graph.analyst_execution import sync_analyst_tracker_from_chunk

        sync_analyst_tracker_from_chunk(wall_time_tracker, chunk)

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def run_analysis(checkpoint: bool = False):
    from cli.stats_handler import StatsCallbackHandler
    from tradingagents.graph.analyst_execution import (
        AnalystWallTimeTracker,
        build_analyst_execution_plan,
        get_initial_analyst_node,
    )
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    # First get all user selections
    selections = get_user_selections()

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    config["checkpoint_enabled"] = checkpoint

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(
        selected_analyst_keys,
        concurrency_limit=config["analyst_concurrency_limit"],
    )
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        first_analyst = get_initial_analyst_node(analyst_execution_plan)
        message_buffer.update_agent_status(first_analyst, "in_progress")
        analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            asset_type=selections["asset_type"],
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process all messages in chunk, deduplicating by message ID
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in message_buffer._processed_message_ids:
                        continue
                    message_buffer._processed_message_ids.add(msg_id)

                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(
                message_buffer,
                chunk,
                wall_time_tracker=analyst_wall_time_tracker,
            )

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge:
                    if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # Streamed chunks are per-node deltas, not full state. Merge them
        # so every report field populated across the run is present.
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


@app.command("broker-status")
def broker_status(
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        min=1,
        max=100,
        help="Number of recent Alpaca orders to show.",
    ),
    status: str = typer.Option(
        "all",
        "--status",
        help="Order status filter: open, closed, or all.",
    ),
):
    """Show Alpaca paper account, current positions, and recent orders."""
    from tradingagents.execution.alpaca_executor import AlpacaExecutor

    executor = AlpacaExecutor()
    account = executor.get_account_info()
    if account is None:
        console.print("[red]Could not fetch Alpaca account info. Check your .env keys.[/red]")
        raise typer.Exit(1)

    account_table = Table(title="Alpaca Paper Account", box=box.SIMPLE)
    account_table.add_column("Metric", style="cyan")
    account_table.add_column("Value", justify="right")
    account_table.add_row("Cash", f"${account['cash']:,.2f}")
    account_table.add_row("Portfolio Value", f"${account['portfolio_value']:,.2f}")
    account_table.add_row("Buying Power", f"${account['buying_power']:,.2f}")
    account_table.add_row("Day Trades", str(account["day_trade_count"]))
    console.print(account_table)

    positions = executor.get_portfolio()
    positions_table = Table(title="Open Positions", box=box.SIMPLE)
    positions_table.add_column("Ticker", style="cyan")
    positions_table.add_column("Qty", justify="right")
    positions_table.add_column("Market Value", justify="right")
    positions_table.add_column("Cost Basis", justify="right")
    positions_table.add_column("Unrealized P/L", justify="right")
    if positions:
        for position in positions:
            positions_table.add_row(
                position["ticker"],
                f"{position['qty']:,.6f}",
                f"${position['market_value']:,.2f}",
                f"${position['cost_basis']:,.2f}",
                f"${position['unrealized_pl']:,.2f}",
            )
    else:
        positions_table.add_row("-", "-", "-", "-", "-")
    console.print(positions_table)

    orders = executor.get_recent_orders(limit=limit, status=status)
    orders_table = Table(title=f"Recent Orders ({status})", box=box.SIMPLE)
    orders_table.add_column("Ticker", style="cyan")
    orders_table.add_column("Side")
    orders_table.add_column("Status")
    orders_table.add_column("Notional", justify="right")
    orders_table.add_column("Filled Qty", justify="right")
    orders_table.add_column("Avg Price", justify="right")
    orders_table.add_column("Submitted")
    if orders:
        for order in orders:
            orders_table.add_row(
                order["ticker"],
                order["side"],
                order["status"],
                f"${order['notional']:,.2f}",
                f"{order['filled_qty']:,.6f}",
                f"${order['filled_avg_price']:,.2f}",
                str(order.get("submitted_at") or "-"),
            )
    else:
        orders_table.add_row("-", "-", "-", "-", "-", "-", "-")
    console.print(orders_table)


def _parse_ticker_csv(tickers: Optional[str]) -> list[str] | None:
    if not tickers:
        return None
    parsed = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
    return parsed or None


def _latest_price(ticker: str) -> float | None:
    try:
        import yfinance as yf

        data = yf.download(
            ticker,
            period="5d",
            progress=False,
            auto_adjust=False,
        )
        if data.empty or "Close" not in data:
            return None
        value = data["Close"].iloc[-1]
        if hasattr(value, "iloc"):
            value = value.iloc[-1]
        return float(value)
    except Exception:
        return None


def _benchmark_return_since(ticker: str, started_at: str | None) -> float | None:
    if not started_at:
        return None
    try:
        import yfinance as yf

        start = datetime.datetime.fromisoformat(started_at).date().isoformat()
        end = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        if data.empty or "Close" not in data or len(data["Close"].dropna()) < 2:
            return None
        close = data["Close"].dropna()
        return (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
    except Exception:
        return None


@app.command("strategy-status")
def strategy_status(
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        min=1,
        max=100,
        help="Number of recent strategy trades/cycles to show.",
    ),
    prices: bool = typer.Option(
        True,
        "--prices/--no-prices",
        help="Fetch latest yfinance prices for strategy P&L estimates.",
    ),
):
    """Show local strategy DB state and broker-vs-strategy mismatches."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.execution.alpaca_executor import AlpacaExecutor
    from tradingagents.state_store import StrategyStateStore

    store = StrategyStateStore(DEFAULT_CONFIG.get("data_cache_dir", "data") + "/strategy_state.db")
    executor = AlpacaExecutor()

    strategy_positions = store.active_positions()
    broker_positions = executor.get_portfolio()
    broker_by_ticker = {position["ticker"]: position for position in broker_positions}
    strategy_by_ticker = {position["ticker"]: position for position in strategy_positions}

    positions_table = Table(title="Strategy DB Positions", box=box.SIMPLE)
    positions_table.add_column("Ticker", style="cyan")
    positions_table.add_column("Status")
    positions_table.add_column("Mode")
    positions_table.add_column("Entry Date")
    positions_table.add_column("Entry", justify="right")
    positions_table.add_column("Qty", justify="right")
    positions_table.add_column("Broker Qty", justify="right")
    positions_table.add_column("P&L", justify="right")
    positions_table.add_column("Stop Order")
    positions_table.add_column("Last Reconciled")
    if strategy_positions:
        for position in strategy_positions:
            latest = _latest_price(position["ticker"]) if prices else None
            qty = float(position.get("quantity") or 0)
            entry = float(position.get("entry_price") or 0)
            pnl = (latest - entry) * qty if latest is not None and entry > 0 else None
            positions_table.add_row(
                str(position["ticker"]),
                str(position["status"]),
                str(position["mode"]),
                str(position.get("entry_date") or "-"),
                f"${entry:,.2f}",
                f"{qty:,.6f}",
                f"{float(position.get('broker_quantity') or 0):,.6f}",
                f"${pnl:,.2f}" if pnl is not None else "-",
                str(position.get("stop_order_id") or "-"),
                str(position.get("last_reconciled_at") or "-"),
            )
    else:
        positions_table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
    console.print(positions_table)

    mismatch_table = Table(title="Broker vs Strategy Diff", box=box.SIMPLE)
    mismatch_table.add_column("Ticker", style="cyan")
    mismatch_table.add_column("Strategy Qty", justify="right")
    mismatch_table.add_column("Broker Qty", justify="right")
    mismatch_table.add_column("Issue")
    mismatches = []
    for ticker in sorted(set(strategy_by_ticker) | set(broker_by_ticker)):
        strategy_qty = float(strategy_by_ticker.get(ticker, {}).get("quantity") or 0)
        broker_qty = float(broker_by_ticker.get(ticker, {}).get("qty") or 0)
        if abs(strategy_qty - broker_qty) > 0.0001:
            if ticker not in broker_by_ticker:
                issue = "strategy only"
            elif ticker not in strategy_by_ticker:
                issue = "broker only"
            else:
                issue = "quantity mismatch"
            mismatches.append((ticker, strategy_qty, broker_qty, issue))
            mismatch_table.add_row(
                ticker,
                f"{strategy_qty:,.6f}",
                f"{broker_qty:,.6f}",
                issue,
            )
    if not mismatches:
        mismatch_table.add_row("-", "-", "-", "no mismatch")
    console.print(mismatch_table)

    cooldowns_table = Table(title="Active Cooldowns", box=box.SIMPLE)
    cooldowns_table.add_column("Ticker", style="cyan")
    cooldowns_table.add_column("Cooldown Until")
    cooldowns_table.add_column("Reason")
    cooldowns = store.active_cooldowns()
    if cooldowns:
        for cooldown in cooldowns:
            cooldowns_table.add_row(
                str(cooldown["ticker"]),
                str(cooldown["cooldown_until"]),
                str(cooldown.get("reason") or "-"),
            )
    else:
        cooldowns_table.add_row("-", "-", "-")
    console.print(cooldowns_table)

    cycles_table = Table(title=f"Recent Strategy Cycles ({limit})", box=box.SIMPLE)
    cycles_table.add_column("ID", justify="right", style="cyan")
    cycles_table.add_column("Mode")
    cycles_table.add_column("Status")
    cycles_table.add_column("Tickers", justify="right")
    cycles_table.add_column("Actions")
    cycles_table.add_column("Started")
    cycles = store.recent_cycles(limit=limit)
    for cycle in cycles:
        cycles_table.add_row(
            str(cycle["cycle_id"]),
            str(cycle["mode"]),
            str(cycle["status"]),
            str(cycle["ticker_count"]),
            str(cycle["action_summary"]),
            str(cycle["started_at"]),
        )
    if not cycles:
        cycles_table.add_row("-", "-", "-", "-", "-", "-")
    console.print(cycles_table)

    trades_table = Table(title=f"Recent Strategy Trades ({limit})", box=box.SIMPLE)
    trades_table.add_column("ID", justify="right", style="cyan")
    trades_table.add_column("Ticker")
    trades_table.add_column("Side")
    trades_table.add_column("Qty", justify="right")
    trades_table.add_column("Price", justify="right")
    trades_table.add_column("Reason")
    trades_table.add_column("Mode")
    trades_table.add_column("Time")
    trades = store.recent_trades(limit=limit)
    if trades:
        for trade in trades:
            trades_table.add_row(
                str(trade["trade_id"]),
                str(trade["ticker"]),
                str(trade["side"]),
                f"{float(trade['quantity']):,.6f}",
                f"${float(trade['fill_price']):,.2f}",
                str(trade.get("reason") or "-"),
                str(trade["mode"]),
                str(trade["timestamp"]),
            )
    else:
        trades_table.add_row("-", "-", "-", "-", "-", "-", "-", "-")
    console.print(trades_table)


@app.command("scorecard")
def scorecard_command(
    resolve: bool = typer.Option(
        False,
        "--resolve",
        help="Try to resolve due outcomes from yfinance before printing.",
    ),
):
    """Show AI decision scorecard performance and allowed position size."""
    from pathlib import Path

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.risk.scorecard import Scorecard
    from tradingagents.scheduler.runner import _scorecard_strategy_key

    scorecard = Scorecard(
        Path(DEFAULT_CONFIG.get("data_cache_dir", "data")) / "scorecard.db",
        horizon_days=int(DEFAULT_CONFIG.get("scorecard_horizon_days", 10)),
        stop_loss_pct=float(DEFAULT_CONFIG.get("scorecard_stop_loss_pct", -0.05)),
        warmup_position_pct=float(DEFAULT_CONFIG.get("scorecard_warmup_position_pct", 0.005)),
        tier1_position_pct=float(DEFAULT_CONFIG.get("scorecard_tier1_position_pct", 0.01)),
        tier2_position_pct=float(DEFAULT_CONFIG.get("scorecard_tier2_position_pct", 0.02)),
        min_resolved_decisions=int(DEFAULT_CONFIG.get("scorecard_min_resolved_decisions", 30)),
        tier2_min_decisions=int(DEFAULT_CONFIG.get("scorecard_tier2_min_decisions", 60)),
        benchmark_ticker=DEFAULT_CONFIG.get("benchmark_ticker") or "SPY",
    )
    if resolve:
        resolved = scorecard.resolve_due_outcomes()
        console.print(f"[green]Resolved {resolved} due scorecard outcome(s).[/green]")

    strategy_key = _scorecard_strategy_key()
    rows = scorecard.summaries()
    if not rows:
        rows = [scorecard.strategy_summary(strategy_key)]

    table = Table(title="AI Decision Scorecard", box=box.SIMPLE)
    table.add_column("Strategy", style="cyan")
    table.add_column("Decisions", justify="right")
    table.add_column("Pending", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Alpha", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Allowed Size", justify="right")
    table.add_column("Status")
    for row in rows:
        gate = scorecard.gate_for_strategy(str(row["strategy_key"]))
        table.add_row(
            str(row["strategy_key"]),
            str(row["resolved_decisions"]),
            str(row["pending_decisions"]),
            f"{float(row['win_rate_pct']):.1f}%",
            f"{float(row['avg_alpha_pct']):+.2f}%",
            f"{float(row['max_drawdown_pct']):.2f}%",
            f"{gate.allowed_position_pct:.2%}",
            gate.status,
        )
    console.print(table)

    leaderboard = Table(title="Champion / Challenger", box=box.SIMPLE)
    leaderboard.add_column("Rank", justify="right")
    leaderboard.add_column("Role")
    leaderboard.add_column("Strategy", style="cyan")
    leaderboard.add_column("Resolved", justify="right")
    leaderboard.add_column("Avg Alpha", justify="right")
    for row in scorecard.leaderboard():
        leaderboard.add_row(
            str(row["rank"]),
            str(row["role"]),
            str(row["strategy_key"]),
            str(row["resolved_decisions"]),
            f"{float(row['avg_alpha_pct']):+.2f}%",
        )
    console.print(leaderboard)


@app.command("experiments")
def experiments_command(
    limit: int = typer.Option(10, "--limit", min=1, max=100),
):
    """Show reproducible validation runs stored in the scorecard database."""
    from tradingagents.risk.scorecard import Scorecard

    scorecard = Scorecard(
        Path(DEFAULT_CONFIG.get("data_cache_dir", "data")) / "scorecard.db"
    )
    table = Table(title="Validation Experiments", box=box.SIMPLE)
    table.add_column("ID", justify="right")
    table.add_column("Kind")
    table.add_column("Strategy", style="cyan")
    table.add_column("Alpha", justify="right")
    table.add_column("Created")
    table.add_column("Artifact")
    experiments = scorecard.experiments(limit)
    for experiment in experiments:
        metrics = experiment["metrics"]
        table.add_row(
            str(experiment["id"]),
            str(experiment["kind"]),
            str(experiment["strategy_key"]),
            f"{float(metrics.get('alpha_pct', 0)):+.2f}%",
            str(experiment["created_at"]),
            str(experiment.get("artifact_path") or "-"),
        )
    if not experiments:
        table.add_row("-", "-", "-", "-", "-", "-")
    console.print(table)


@app.command("decision-replay")
def decision_replay_command(
    decision_id: Optional[int] = typer.Option(None, "--decision-id", min=1),
    ticker: Optional[str] = typer.Option(
        None, "--ticker", help="Inspect this ticker's latest decision."
    ),
):
    """Inspect one stored AI decision and its evidence without live API calls."""
    from tradingagents.risk.scorecard import Scorecard

    scorecard = Scorecard(
        Path(DEFAULT_CONFIG.get("data_cache_dir", "data")) / "scorecard.db"
    )
    if (decision_id is None) == (ticker is None):
        console.print("[red]Provide exactly one of --decision-id or --ticker.[/red]")
        raise typer.Exit(2)
    if ticker is not None:
        decisions = scorecard.decisions_for_ticker(ticker)
        if not decisions:
            console.print(f"[red]No stored decisions were found for {ticker.upper()}.[/red]")
            raise typer.Exit(1)
        decision_id = int(decisions[-1]["id"])
    assert decision_id is not None
    decision = scorecard.decision_artifact(decision_id)
    if decision is None:
        console.print(f"[red]Decision {decision_id} was not found.[/red]")
        raise typer.Exit(1)

    summary = Table(title=f"Decision {decision_id}", box=box.SIMPLE)
    summary.add_column("Field", style="cyan")
    summary.add_column("Value")
    for field in (
        "strategy_key",
        "ticker",
        "trade_date",
        "rating",
        "mode",
        "model_provider",
        "quick_model",
        "deep_model",
        "entry_price",
        "resolved_at",
        "directional_alpha_pct",
    ):
        summary.add_row(field.replace("_", " "), str(decision.get(field) or "-"))
    console.print(summary)
    console.print(Panel(str(decision.get("final_trade_decision") or "-"), title="Final Decision"))
    for name, report in decision["evidence"].items():
        console.print(Panel(str(report or "-"), title=name.replace("_", " ").title()))


@app.command("health")
def health_command():
    """Show autonomous-trader reconciliation, protection, risk, and performance health."""
    from tradingagents.state_store import StrategyStateStore

    store = StrategyStateStore(DEFAULT_CONFIG.get("data_cache_dir", "data") + "/strategy_state.db")
    health = store.health_snapshot()
    table = Table(title="Autonomous Trader Health", box=box.SIMPLE)
    table.add_column("Check", style="cyan")
    table.add_column("Value")
    last_cycle = health.get("last_cycle") or {}
    risk = health.get("risk") or {}
    performance = health.get("performance") or {}
    benchmark = DEFAULT_CONFIG.get("benchmark_ticker") or "SPY"
    benchmark_return = _benchmark_return_since(
        benchmark, performance.get("first_equity_at")
    )
    table.add_row("Last Cycle", str(last_cycle.get("completed_at") or last_cycle.get("started_at") or "never"))
    table.add_row("Last Cycle Status", str(last_cycle.get("status") or "unknown"))
    table.add_row("Last Reconciliation", str(health.get("last_reconciled_at") or "never"))
    table.add_row("Open Positions", str(health.get("open_positions", 0)))
    table.add_row("Active Orders", str(health.get("active_orders", 0)))
    table.add_row("Unprotected", ", ".join(health.get("unprotected_tickers") or []) or "none")
    table.add_row("Risk Halt", str(bool(risk.get("manual_halt") or risk.get("halted_until"))))
    table.add_row("Risk Reason", str(risk.get("halt_reason") or "none"))
    table.add_row("Confirmed Fills", str(performance.get("fill_count", 0)))
    table.add_row("Realized P&L", f"${float(performance.get('realized_pnl', 0)):,.2f}")
    table.add_row("Days Tracked", str(performance.get("tracking_days") or "not enough data"))
    account_return = performance.get("account_return_pct")
    table.add_row(
        "Whole Account Return",
        f"{float(account_return):+.2f}%" if account_return is not None else "not enough data",
    )
    table.add_row(
        f"{benchmark} Return (same dates)",
        f"{benchmark_return:+.2f}%" if benchmark_return is not None else "market data unavailable",
    )
    table.add_row(
        f"Whole Account Alpha vs {benchmark}",
        f"{float(account_return) - benchmark_return:+.2f}%"
        if account_return is not None and benchmark_return is not None
        else "not enough data",
    )
    max_drawdown = performance.get("max_account_drawdown_pct")
    table.add_row(
        "Maximum Account Drawdown",
        f"{float(max_drawdown):.2f}%" if max_drawdown is not None else "not enough data",
    )
    deployed = performance.get("capital_deployed_pct")
    table.add_row(
        "Capital Currently Deployed",
        f"{float(deployed):.2f}%" if deployed is not None else "not enough data",
    )
    realized_return = performance.get("realized_return_on_capital_pct")
    table.add_row(
        "Closed-Trade Return on Capital",
        f"{float(realized_return):+.2f}%" if realized_return is not None else "no closed trades",
    )
    closed_trades = int(performance.get("closed_trade_count", 0))
    table.add_row("Closed Trades", str(closed_trades))
    win_rate = performance.get("win_rate_pct")
    table.add_row(
        "Closed-Trade Win Rate",
        f"{float(win_rate):.1f}%" if win_rate is not None else "no closed trades",
    )
    average_win = performance.get("average_win_pct")
    table.add_row(
        "Average Winner",
        f"{float(average_win):+.2f}%" if average_win is not None else "no winners",
    )
    average_loss = performance.get("average_loss_pct")
    table.add_row(
        "Average Loser",
        f"{float(average_loss):+.2f}%" if average_loss is not None else "no losses",
    )
    profit_factor = performance.get("profit_factor")
    table.add_row(
        "Profit Factor (gross)",
        f"{float(profit_factor):.2f}" if profit_factor is not None else "needs a losing trade",
    )
    console.print(table)


@app.command("halt-trading")
def halt_trading_command(
    reason: str = typer.Option("manual operator halt", "--reason", help="Audit reason for the halt."),
):
    """Persistently disable broker-backed new trading."""
    from tradingagents.state_store import StrategyStateStore

    store = StrategyStateStore(DEFAULT_CONFIG.get("data_cache_dir", "data") + "/strategy_state.db")
    store.set_manual_halt(reason)
    console.print(f"[red]Trading halted:[/red] {reason}")


@app.command("resume-trading")
def resume_trading_command(
    confirmation: str = typer.Option(..., "--confirmation", help="Enter RESUME TRADING exactly."),
):
    """Clear a persistent manual/risk halt after the cause has been reviewed."""
    if confirmation != "RESUME TRADING":
        console.print("[red]Confirmation must be exactly: RESUME TRADING[/red]")
        raise typer.Exit(2)
    from tradingagents.state_store import StrategyStateStore

    store = StrategyStateStore(DEFAULT_CONFIG.get("data_cache_dir", "data") + "/strategy_state.db")
    store.clear_manual_halt()
    console.print("[green]Trading halt cleared.[/green]")


@app.command("acknowledge-health")
def acknowledge_health_command(
    note: str = typer.Option(..., "--note", help="What was reviewed and corrected."),
):
    """Acknowledge reviewed health events while preserving their audit history."""
    from tradingagents.state_store import StrategyStateStore

    store = StrategyStateStore(DEFAULT_CONFIG.get("data_cache_dir", "data") + "/strategy_state.db")
    count = store.acknowledge_health_events(note)
    console.print(f"[green]Acknowledged {count} health event(s).[/green]")


@app.command("release-audit")
def release_audit_command():
    """Generate the machine-checked report required before real mode can unlock."""
    from tradingagents.risk.release_gate import build_release_report, release_strategy_config
    from tradingagents.risk.scorecard import Scorecard
    from tradingagents.scheduler.runner import _scorecard_strategy_key
    from tradingagents.state_store import StrategyStateStore

    strategy_key = _scorecard_strategy_key()
    store = StrategyStateStore(DEFAULT_CONFIG.get("data_cache_dir", "data") + "/strategy_state.db")
    scorecard = Scorecard(
        Path(DEFAULT_CONFIG.get("data_cache_dir", "data")) / "scorecard.db"
    )
    output_path = Path(str(DEFAULT_CONFIG.get("real_money_validation_report")))
    report = build_release_report(
        store=store,
        scorecard=scorecard,
        strategy_key=strategy_key,
        account_id=str(DEFAULT_CONFIG.get("expected_real_account_id", "")),
        output_path=output_path,
        strategy_config=release_strategy_config(DEFAULT_CONFIG),
        backtest_validation_path=DEFAULT_CONFIG.get("backtest_validation_report"),
    )
    table = Table(title="Real-Money Release Audit", box=box.SIMPLE)
    table.add_column("Gate", style="cyan")
    table.add_column("Passed", justify="center")
    for name, passed in report["checks"].items():
        table.add_row(name.replace("_", " "), "yes" if passed else "NO")
    console.print(table)
    console.print(f"Report: {output_path}")
    if not report["approved"]:
        console.print("[red]Real-money mode remains locked.[/red]")
        raise typer.Exit(1)
    console.print("[green]Release gates passed. Real mode still requires all runtime locks.[/green]")


@app.command("backup-state")
def backup_state_command(
    output_dir: Path = typer.Option(
        Path("backups"), "--output-dir", help="Directory for verified SQLite backups."
    ),
):
    """Create and integrity-check a backup of the autonomous-trader ledger."""
    from tradingagents.state_store import StrategyStateStore

    source = Path(DEFAULT_CONFIG.get("data_cache_dir", "data")) / "strategy_state.db"
    store = StrategyStateStore(source)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_dir / f"strategy-state-{timestamp}.db"
    store.backup_to(target)
    console.print(f"[green]Verified backup created:[/green] {target.resolve()}")


@app.command("run-cycle")
def run_cycle_command(
    mode: str = typer.Option(
        "dry-run",
        "--mode",
        help="Execution mode: dry-run, shadow, paper, or real. 'live' is a safe paper alias.",
    ),
    tickers: Optional[str] = typer.Option(
        None,
        "--tickers",
        help="Comma-separated ticker list. Manual tickers are allowed by default.",
    ),
    discover: bool = typer.Option(
        False,
        "--discover",
        help="Keep manual tickers and add automatically screened candidates.",
    ),
    confirm_real_money: Optional[str] = typer.Option(
        None,
        "--confirm-real-money",
        help="Exact one-time confirmation phrase required only for real mode.",
    ),
):
    """Run one background trading cycle."""
    from tradingagents.scheduler.runner import run_cycle

    try:
        summary = run_cycle(
            tickers=_parse_ticker_csv(tickers),
            discover=discover,
            mode=mode,
            real_money_confirmation=confirm_real_money,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    if not summary:
        console.print("[yellow]Cycle did not return a summary.[/yellow]")
        return

    table = Table(title="Cycle Summary", box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Mode", str(summary.get("mode")))
    table.add_row("Status", str(summary.get("status")))
    table.add_row("Tickers", str(summary.get("tickers", 0)))
    table.add_row("Symbols", ", ".join(summary.get("ticker_symbols") or []) or "-")
    table.add_row("Signals", str(summary.get("signals", 0)))
    table.add_row("Orders Submitted", str(summary.get("submitted", 0)))
    table.add_row("Confirmed Fill Events", str(summary.get("executed", 0)))
    table.add_row("Simulated", str(summary.get("simulated", 0)))
    table.add_row("Analysis Failures", str(summary.get("analysis_failures", 0)))
    if summary.get("reason"):
        table.add_row("Reason", str(summary["reason"]))
    console.print(table)


@app.command("run-bot")
def run_bot_command(
    mode: str = typer.Option(
        "dry-run",
        "--mode",
        help="Execution mode: dry-run, shadow, paper, or real. 'live' is a safe paper alias.",
    ),
    interval: Optional[int] = typer.Option(
        None,
        "--interval",
        min=1,
        help="Minutes between cycles. Overrides the weekday schedule.",
    ),
    daily_at: str = typer.Option(
        "08:45",
        "--daily-at",
        help="Weekday run time in 24-hour America/Chicago time.",
    ),
    tickers: Optional[str] = typer.Option(
        None,
        "--tickers",
        help="Comma-separated ticker list. Manual tickers are allowed by default.",
    ),
    discover: bool = typer.Option(
        False,
        "--discover",
        help="Keep manual tickers and add automatically screened candidates.",
    ),
    run_now: bool = typer.Option(
        True,
        "--run-now/--wait-first",
        help="Run one cycle at startup instead of waiting for the first interval.",
    ),
    log_file: Path = typer.Option(
        Path("logs/trading-bot.log"),
        "--log-file",
        help="Rotating log file for unattended runs.",
    ),
    confirm_real_money: Optional[str] = typer.Option(
        None,
        "--confirm-real-money",
        help="Exact one-time confirmation phrase required only for real mode.",
    ),
):
    """Run the bot continuously until stopped."""
    from tradingagents.scheduler.runner import start_scheduler

    resolved_log = _configure_bot_logging(log_file)
    schedule_text = (
        f"every {interval}m" if interval is not None else f"weekdays at {daily_at} CT"
    )
    console.print(
        f"[cyan]Starting bot[/cyan]: mode={mode}, schedule={schedule_text}, "
        f"log={resolved_log}. Press Ctrl+C to stop."
    )
    try:
        start_scheduler(
            interval_minutes=interval,
            mode=mode,
            tickers=_parse_ticker_csv(tickers),
            discover=discover,
            run_immediately=run_now,
            daily_at=daily_at,
            real_money_confirmation=confirm_real_money,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot stopped cleanly.[/yellow]")


@app.command("walk-forward")
def walk_forward_command(
    ticker: str = typer.Option(
        ...,
        "--ticker",
        help="Ticker to test.",
    ),
    start: str = typer.Option(
        "2020-01-01",
        "--start",
        help="Start date in YYYY-MM-DD format.",
    ),
    end: str = typer.Option(
        datetime.date.today().isoformat(),
        "--end",
        help="End date in YYYY-MM-DD format.",
    ),
    position_pct: float = typer.Option(
        0.02,
        "--position-pct",
        min=0.001,
        max=0.10,
        help="Fraction of equity used per entry.",
    ),
    min_score: float = typer.Option(
        2.0,
        "--min-score",
        min=1.0,
        max=4.0,
        help="Minimum past-data signal score required for entry.",
    ),
    stop_loss_pct: float = typer.Option(
        -0.05,
        "--stop-loss-pct",
        min=-0.50,
        max=-0.001,
        help="Stop loss as a decimal return.",
    ),
    take_profit_pct: float = typer.Option(
        0.08,
        "--take-profit-pct",
        min=0.001,
        max=1.0,
        help="Take profit as a decimal return.",
    ),
):
    """Run a no-foresight historical walk-forward simulation."""
    import yfinance as yf

    from backtests.walk_forward import WalkForwardConfig, run_walk_forward_backtest

    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    if data.empty:
        console.print(f"[red]No data returned for {ticker}.[/red]")
        raise typer.Exit(1)
    data = data.reset_index()

    result = run_walk_forward_backtest(
        data,
        WalkForwardConfig(
            position_pct=position_pct,
            min_score=min_score,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        ),
    )

    table = Table(title=f"Walk-Forward Backtest: {ticker.upper()}", box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total Return", f"{result['total_return_pct']:.2f}%")
    table.add_row("Max Drawdown", f"{result['max_drawdown_pct']:.2f}%")
    table.add_row("Sharpe", f"{result['sharpe']:.2f}")
    table.add_row("Win Rate", f"{result['win_rate_pct']:.2f}%")
    table.add_row("Trades", str(result["num_trades"]))
    table.add_row("Final Equity", f"${result['final_equity']:,.2f}")
    console.print(table)


@app.command("data-audit")
def data_audit_command(
    ticker: str = typer.Option(
        ..., "--ticker", help="Ticker whose OHLCV history should be checked."
    ),
    start: str = typer.Option(
        "2020-01-01", "--start", help="Start date in YYYY-MM-DD format."
    ),
    end: str = typer.Option(
        datetime.date.today().isoformat(), "--end", help="End date in YYYY-MM-DD format."
    ),
):
    """Check market data health and prove feature calculations do not use future rows."""
    import yfinance as yf

    from backtests.data_audit import audit_feature_lookahead, audit_market_data
    from backtests.walk_forward import build_walk_forward_features

    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        ).reset_index()
    except Exception as exc:
        console.print(f"[red]Could not download {ticker.upper()}: {exc}[/red]")
        raise typer.Exit(1) from exc

    health = audit_market_data(data)
    lookahead = audit_feature_lookahead(data, build_walk_forward_features)
    table = Table(title=f"Data Audit: {ticker.upper()}", box=box.SIMPLE)
    table.add_column("Check", style="cyan")
    table.add_column("Result", justify="right")
    table.add_row("Rows", str(health["rows"]))
    table.add_row("Range", f"{health['start'] or '-'} to {health['end'] or '-'}")
    table.add_row("OHLCV health", "PASS" if health["ok"] else "FAIL")
    table.add_row("Prefix lookahead test", "PASS" if lookahead["ok"] else "FAIL")
    table.add_row("Feature comparisons", str(lookahead["checks"]))
    console.print(table)
    for issue in [*health["issues"], *lookahead["issues"]]:
        color = "red" if issue["severity"] == "error" else "yellow"
        console.print(
            f"[{color}]{issue['severity'].upper()} {issue['code']}: "
            f"{issue['message']}[/{color}]"
        )
    if not health["ok"] or not lookahead["ok"]:
        raise typer.Exit(1)


@app.command("ml-shadow")
def ml_shadow_command(
    tickers: str = typer.Option(
        ..., "--tickers", help="Comma-separated stock tickers."
    ),
    start: str = typer.Option(
        "2020-01-01", "--start", help="Start date in YYYY-MM-DD format."
    ),
    end: str = typer.Option(
        datetime.date.today().isoformat(), "--end", help="End date in YYYY-MM-DD format."
    ),
    horizon_days: int = typer.Option(
        10, "--horizon-days", min=1, max=60, help="Trading bars between entry and label."
    ),
    database: Path = typer.Option(
        Path("data/ml_shadow.db"), "--database", help="Local shadow-ledger SQLite path."
    ),
):
    """Build leakage-safe ML examples without changing trading decisions."""
    import yfinance as yf

    from backtests.data_audit import audit_market_data
    from backtests.ml_shadow import MLShadowLedger, build_ml_samples

    symbols = _parse_ticker_csv(tickers)
    if not symbols:
        console.print("[red]Provide at least one ticker.[/red]")
        raise typer.Exit(2)

    try:
        benchmark = yf.download(
            "SPY",
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        ).reset_index()
    except Exception as exc:
        console.print(f"[red]Could not download SPY: {exc}[/red]")
        raise typer.Exit(1) from exc
    if not audit_market_data(benchmark)["ok"]:
        console.print("[red]SPY data failed the health audit; no samples were written.[/red]")
        raise typer.Exit(1)

    ledger = MLShadowLedger(database)
    table = Table(title="ML Shadow Dataset", box=box.SIMPLE)
    table.add_column("Ticker", style="cyan")
    table.add_column("Built", justify="right")
    table.add_column("New", justify="right")
    failures = 0
    for symbol in symbols:
        try:
            stock = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            ).reset_index()
            audit = audit_market_data(stock)
            if not audit["ok"]:
                failures += 1
                table.add_row(symbol, "FAILED AUDIT", "0")
                continue
            samples = build_ml_samples(
                stock, benchmark, ticker=symbol, horizon_days=horizon_days
            )
            inserted = ledger.record(samples)
            table.add_row(symbol, str(len(samples)), str(inserted))
        except Exception as exc:
            failures += 1
            table.add_row(symbol, f"FAILED: {exc}", "0")
    console.print(table)
    summary = ledger.summary()
    console.print(
        f"Ledger: {summary['samples']} samples across {summary['tickers']} tickers | "
        f"{summary['start_date'] or '-'} to {summary['end_date'] or '-'} | "
        f"outperformed SPY {summary['outperform_rate_pct']:.1f}%"
    )
    console.print(
        "[dim]Shadow-only: this database is not read by the scheduler or order router.[/dim]"
    )
    if failures == len(symbols):
        raise typer.Exit(1)


@app.command("ml-build-sp500")
def ml_build_sp500_command(
    start: str = typer.Option(
        "2010-01-01", "--start", help="First point-in-time membership date to include."
    ),
    end: str = typer.Option(
        datetime.date.today().isoformat(), "--end", help="Last sample date to include."
    ),
    horizon_days: int = typer.Option(
        10, "--horizon-days", min=1, max=60, help="Trading bars between entry and label."
    ),
    database: Path = typer.Option(
        Path("data/ml_shadow.db"), "--database", help="Local shadow-ledger SQLite path."
    ),
    membership_cache: Path = typer.Option(
        Path("data/sp500_history.csv"),
        "--membership-cache",
        help="Cached point-in-time constituent snapshots.",
    ),
    source_url: str = typer.Option(
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
        "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv",
        "--source-url",
        help="Public date,tickers snapshot CSV; ignored when a cache already exists.",
    ),
    refresh_membership: bool = typer.Option(
        False, "--refresh-membership", help="Refresh the cached membership source."
    ),
    offset: int = typer.Option(
        0, "--offset", min=0, help="Skip this many alphabetized tickers for resumable batches."
    ),
    max_tickers: Optional[int] = typer.Option(
        None, "--max-tickers", min=1, help="Optional batch size; reruns are idempotent."
    ),
):
    """Build a point-in-time historical S&P 500 ML dataset without LLM calls."""
    import yfinance as yf

    from backtests.data_audit import audit_market_data, trim_incomplete_trailing_rows
    from backtests.ml_shadow import MLShadowLedger, build_ml_samples
    from backtests.sp500_history import (
        content_sha256,
        fetch_membership_history,
        filter_samples_for_membership,
        intervals_by_ticker,
        parse_membership_history,
        tickers_overlapping,
        yahoo_ticker,
    )

    try:
        start_date = datetime.date.fromisoformat(start)
        end_date = datetime.date.fromisoformat(end)
    except ValueError as exc:
        console.print(f"[red]Invalid ISO date: {exc}[/red]")
        raise typer.Exit(2) from exc
    if start_date >= end_date:
        console.print("[red]--start must be before --end.[/red]")
        raise typer.Exit(2)

    try:
        content, source = fetch_membership_history(
            source_url=source_url,
            cache_path=membership_cache,
            refresh=refresh_membership,
        )
        grouped = intervals_by_ticker(parse_membership_history(content))
    except Exception as exc:
        console.print(f"[red]Could not load point-in-time membership history: {exc}[/red]")
        raise typer.Exit(1) from exc

    all_symbols = tickers_overlapping(grouped, start_date, end_date)
    symbols = all_symbols[offset : offset + max_tickers if max_tickers else None]
    if not symbols:
        console.print("[red]No historical constituents matched this batch.[/red]")
        raise typer.Exit(2)
    coverage_start = min(item.start_date for values in grouped.values() for item in values)
    coverage_end = max(item.end_date for values in grouped.values() for item in values)
    console.print(
        f"Point-in-time source: {len(all_symbols)} eligible symbols | "
        f"coverage {coverage_start} to {coverage_end} | batch {offset + 1}-"
        f"{offset + len(symbols)}"
    )
    if end_date > coverage_end:
        console.print(
            f"[yellow]Membership source ends {coverage_end}; samples after that date "
            "will not be guessed or mislabeled.[/yellow]"
        )

    download_start = (start_date - datetime.timedelta(days=120)).isoformat()
    try:
        benchmark = _download_yfinance_quiet(
            yf,
            "SPY", start=download_start, end=end, progress=False,
            auto_adjust=True, multi_level_index=False,
        ).reset_index()
    except Exception as exc:
        console.print(f"[red]Could not download SPY: {exc}[/red]")
        raise typer.Exit(1) from exc
    if benchmark.empty:
        console.print(
            "[red]Yahoo returned no SPY history. This is commonly temporary "
            "throttling; wait and rerun the same idempotent batch.[/red]"
        )
        raise typer.Exit(1)
    benchmark, trimmed_benchmark_rows = trim_incomplete_trailing_rows(benchmark)
    benchmark_audit = audit_market_data(benchmark)
    if not benchmark_audit["ok"]:
        console.print(
            f"[red]SPY data failed its health audit; nothing was written. "
            f"Rows={benchmark_audit['rows']}, range={benchmark_audit['start']} to "
            f"{benchmark_audit['end']}.[/red]"
        )
        for issue in benchmark_audit["issues"]:
            console.print(
                f"[red]{issue['code']} ({issue['count']}): {issue['message']}[/red]"
            )
        console.print(
            "[yellow]This is often a temporary Yahoo response. Rerun the same "
            "idempotent command; do not train from a failed build.[/yellow]"
        )
        raise typer.Exit(1)
    if trimmed_benchmark_rows:
        console.print(
            f"[yellow]Removed {trimmed_benchmark_rows} incomplete trailing SPY "
            "row(s), then passed the audit.[/yellow]"
        )

    ledger = MLShadowLedger(database)
    inserted = 0
    succeeded = 0
    failures: list[dict[str, str]] = []
    for index, symbol in enumerate(symbols, start=1):
        try:
            stock = _download_yfinance_quiet(
                yf,
                yahoo_ticker(symbol), start=download_start, end=end, progress=False,
                auto_adjust=True, multi_level_index=False,
            ).reset_index()
            if stock.empty:
                failures.append(
                    {
                        "ticker": symbol,
                        "category": "unavailable_history",
                        "error": "Yahoo returned no historical prices (often delisted/renamed/acquired)",
                    }
                )
                continue
            stock, _ = trim_incomplete_trailing_rows(stock)
            audit = audit_market_data(stock)
            if not audit["ok"]:
                failures.append(
                    {
                        "ticker": symbol,
                        "category": "data_audit",
                        "error": "; ".join(issue["code"] for issue in audit["issues"]),
                    }
                )
                continue
            samples = build_ml_samples(
                stock, benchmark, ticker=symbol, horizon_days=horizon_days
            )
            samples = [
                sample for sample in samples
                if start_date <= datetime.date.fromisoformat(sample["sample_date"]) <= end_date
            ]
            samples = filter_samples_for_membership(samples, grouped[symbol])
            inserted += ledger.record(samples)
            succeeded += 1
        except Exception as exc:
            failures.append(
                {
                    "ticker": symbol,
                    "category": "download_or_processing_error",
                    "error": str(exc),
                }
            )
        if index % 25 == 0 or index == len(symbols):
            console.print(
                f"Processed {index}/{len(symbols)} | successful {succeeded} | "
                f"failed {len(failures)} | new samples {inserted}"
            )

    build_id = ledger.record_build(
        {
            "universe": "sp500-point-in-time",
            "source": source,
            "source_sha256": content_sha256(content),
            "start_date": start,
            "end_date": end,
            "horizon_days": horizon_days,
            "attempted": len(symbols),
            "succeeded": succeeded,
            "failed": len(failures),
            "inserted": inserted,
            "offset": offset,
            "membership_coverage_start": coverage_start.isoformat(),
            "membership_coverage_end": coverage_end.isoformat(),
            "failure_details": failures,
        }
    )
    summary = ledger.summary()
    console.print(
        f"Build {build_id}: ledger now has {summary['samples']} samples across "
        f"{summary['tickers']} tickers. No LLM quota was used."
    )
    if failures:
        from collections import Counter

        failure_counts = Counter(failure["category"] for failure in failures)
        console.print(
            "[yellow]Skipped histories: "
            + ", ".join(
                f"{category}={count}" for category, count in sorted(failure_counts.items())
            )
            + ". Full details were recorded in SQLite.[/yellow]"
        )
        unexpected = [
            failure
            for failure in failures
            if failure["category"] == "download_or_processing_error"
        ]
        for failure in unexpected[:10]:
            console.print(
                f"[red]{failure['ticker']}: {failure['error']}[/red]"
            )
        if len(unexpected) > 10:
            console.print(
                f"[red]...and {len(unexpected) - 10} additional unexpected errors.[/red]"
            )
    console.print("[dim]Shadow-only: neither this dataset nor its model can place orders.[/dim]")
    if succeeded == 0:
        raise typer.Exit(1)


@app.command("ml-train")
def ml_train_command(
    database: Path = typer.Option(
        Path("data/ml_shadow.db"), "--database", help="ML shadow-ledger SQLite path."
    ),
    model_path: Path = typer.Option(
        Path("data/ml_model.json"), "--model", help="Transparent JSON model output."
    ),
    horizon_days: int = typer.Option(
        10, "--horizon-days", min=1, max=60, help="Dataset horizon to train."
    ),
    validation_start: Optional[str] = typer.Option(
        None, "--validation-start", help="ISO date; defaults to two calendar years back."
    ),
    test_start: Optional[str] = typer.Option(
        None, "--test-start", help="ISO date; defaults to one calendar year back."
    ),
    min_samples: int = typer.Option(
        1_000, "--min-samples", min=100, help="Guard against meaningless tiny training sets."
    ),
):
    """Train and evaluate a chronological shadow-only alpha model."""
    from backtests.ml_model import default_split_dates, save_model, train_linear_alpha_model
    from backtests.ml_shadow import FEATURE_VERSION, MLShadowLedger

    ledger = MLShadowLedger(database)
    samples = ledger.load_samples(
        horizon_days=horizon_days, feature_version=FEATURE_VERSION
    )
    if not samples:
        console.print("[red]No matching samples. Run ml-build-sp500 first.[/red]")
        raise typer.Exit(1)
    default_validation, default_test = default_split_dates(samples)
    validation_start = validation_start or default_validation.isoformat()
    test_start = test_start or default_test.isoformat()
    try:
        model = train_linear_alpha_model(
            samples,
            validation_start=validation_start,
            test_start=test_start,
            min_samples=min_samples,
        )
        destination = save_model(model, model_path)
    except Exception as exc:
        console.print(f"[red]Training failed safely: {exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table(title="ML Chronological Evaluation", box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Validation", justify="right")
    table.add_column("Held-out test", justify="right")
    for key, label in (
        ("roc_auc", "ROC AUC"),
        ("brier_score", "Brier score"),
        ("alpha_correlation", "Alpha correlation"),
        ("baseline_mean_alpha_pct", "Baseline alpha %"),
        ("top_decile_mean_alpha_pct", "Top-decile alpha %"),
    ):
        validation_value = model["validation_metrics"][key]
        test_value = model["test_metrics"][key]
        table.add_row(
            label,
            "n/a" if validation_value is None else f"{validation_value:.4f}",
            "n/a" if test_value is None else f"{test_value:.4f}",
        )
    console.print(table)
    counts = model["sample_counts"]
    console.print(
        f"Samples: train {counts['train']}, validation {counts['validation']}, "
        f"test {counts['test']}, boundary-purged {counts['purged']} | "
        f"model: {destination}"
    )
    console.print("[dim]Shadow-only model; it is not loaded anywhere in trade execution.[/dim]")


@app.command("ml-predict")
def ml_predict_command(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated symbols to rank."),
    model_path: Path = typer.Option(
        Path("data/ml_model.json"), "--model", help="JSON model produced by ml-train."
    ),
):
    """Rank current candidates with the ML model without placing orders or using an LLM."""
    import yfinance as yf

    from backtests.ml_model import load_model, predict_feature_rows
    from backtests.ml_shadow import build_ml_feature_rows

    symbols = _parse_ticker_csv(tickers)
    if not symbols:
        console.print("[red]Provide at least one ticker.[/red]")
        raise typer.Exit(2)
    try:
        model = load_model(model_path)
        start = (datetime.date.today() - datetime.timedelta(days=500)).isoformat()
        end = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        benchmark = yf.download(
            "SPY", start=start, end=end, progress=False,
            auto_adjust=True, multi_level_index=False,
        ).reset_index()
        rows = []
        failures = []
        for symbol in symbols:
            try:
                stock = yf.download(
                    symbol, start=start, end=end, progress=False,
                    auto_adjust=True, multi_level_index=False,
                ).reset_index()
                feature_rows = build_ml_feature_rows(stock, benchmark, ticker=symbol)
                if not feature_rows:
                    raise ValueError("not enough aligned price history")
                rows.append(feature_rows[-1])
            except Exception as exc:
                failures.append(f"{symbol}: {exc}")
        predictions = sorted(
            predict_feature_rows(model, rows),
            key=lambda item: item["outperform_probability"],
            reverse=True,
        )
    except Exception as exc:
        console.print(f"[red]Prediction failed safely: {exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table(title="ML Shadow Ranking", box=box.SIMPLE)
    table.add_column("Ticker", style="cyan")
    table.add_column("As of")
    table.add_column("P(outperform SPY)", justify="right")
    table.add_column("Expected alpha", justify="right")
    table.add_column("Shadow signal", justify="right")
    for item in predictions:
        table.add_row(
            item["ticker"], item["sample_date"],
            f"{item['outperform_probability'] * 100:.1f}%",
            f"{item['expected_alpha_pct']:.2f}%",
            "YES" if item["shadow_signal"] else "NO",
        )
    console.print(table)
    for failure in failures:
        console.print(f"[yellow]{failure}[/yellow]")
    console.print("[dim]Advisory shadow output only; no order path reads this result.[/dim]")


@app.command("replay-backtest")
def replay_backtest_command(
    ticker: Optional[str] = typer.Option(
        None, "--ticker", help="One ticker to inspect; not enough for release validation."
    ),
    tickers: Optional[str] = typer.Option(
        None, "--tickers", help="Comma-separated tickers for cross-ticker validation."
    ),
    strategy_key: Optional[str] = typer.Option(
        None, "--strategy-key", help="Optional exact scorecard strategy version to replay."
    ),
    initial_cash: float = typer.Option(
        10_000.0, "--initial-cash", min=100.0, help="Starting simulated cash."
    ),
    output_report: Path = typer.Option(
        Path(str(DEFAULT_CONFIG.get("backtest_validation_report"))),
        "--output-report",
        help="JSON evidence file consumed by the real-money release audit.",
    ),
):
    """Backtest stored graph ratings using the scheduler's actual entry/exit rules."""
    from datetime import timedelta

    import yfinance as yf

    from backtests.strategy_replay import (
        ReplayConfig,
        replay_graph_decisions,
        summarize_replay_results,
    )
    from tradingagents.risk.scorecard import Scorecard
    from tradingagents.scheduler.runner import _scorecard_strategy_key
    from tradingagents.strategy.rules import strategy_rules_from_config

    selected_strategy_key = strategy_key or _scorecard_strategy_key()
    requested_tickers = []
    for value in ([ticker] if ticker else []) + ((tickers or "").split(",")):
        normalized = value.strip().upper()
        if normalized and normalized not in requested_tickers:
            requested_tickers.append(normalized)
    if not requested_tickers:
        console.print("[red]Provide --ticker or --tickers.[/red]")
        raise typer.Exit(2)

    scorecard = Scorecard(
        Path(DEFAULT_CONFIG.get("data_cache_dir", "data")) / "scorecard.db"
    )
    results = {}
    stored_decisions = 0
    rules = strategy_rules_from_config(DEFAULT_CONFIG)
    for symbol in requested_tickers:
        decisions = scorecard.decisions_for_ticker(
            symbol,
            strategy_key=selected_strategy_key,
        )
        if not decisions:
            console.print(f"[yellow]Skipping {symbol}: no stored decisions.[/yellow]")
            continue
        start = (
            datetime.datetime.fromisoformat(decisions[0]["trade_date"]).date()
            - timedelta(days=7)
        )
        end = (
            datetime.datetime.fromisoformat(decisions[-1]["trade_date"]).date()
            + timedelta(days=30)
        )
        prices = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
        if prices.empty:
            console.print(f"[yellow]Skipping {symbol}: no price history.[/yellow]")
            continue
        results[symbol] = replay_graph_decisions(
            prices.reset_index(),
            decisions,
            rules=rules,
            config=ReplayConfig(initial_cash=initial_cash),
        )
        stored_decisions += len(decisions)
    if not results:
        console.print("[red]No requested ticker had both decisions and price history.[/red]")
        raise typer.Exit(1)
    result = summarize_replay_results(results)
    import json

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "strategy_key": selected_strategy_key,
                "stored_decisions": stored_decisions,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    experiment_id = scorecard.record_experiment(
        kind="graph-decision-replay",
        strategy_key=selected_strategy_key,
        config={
            "initial_cash": initial_cash,
            "rules": rules.__dict__,
            "scorecard_size_cap": ReplayConfig().scorecard_size_cap,
        },
        data={
            "tickers": result["tickers"],
            "start_date": result["start_date"],
            "end_date": result["end_date"],
            "stored_decisions": stored_decisions,
        },
        metrics={
            key: result[key]
            for key in (
                "ticker_count",
                "num_trades",
                "total_return_pct",
                "benchmark_return_pct",
                "alpha_pct",
                "max_drawdown_pct",
            )
        },
        artifact_path=str(output_report.resolve()),
    )
    table = Table(title="Cross-Ticker Graph Decision Replay", box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Tickers", ", ".join(result["tickers"]))
    table.add_row("Stored Decisions", str(stored_decisions))
    table.add_row("Completed Trades", str(result["num_trades"]))
    table.add_row("Total Return", f"{result['total_return_pct']:.2f}%")
    table.add_row("Buy-and-Hold Return", f"{result['benchmark_return_pct']:.2f}%")
    table.add_row("Alpha", f"{result['alpha_pct']:+.2f}%")
    table.add_row("Worst Drawdown", f"{result['max_drawdown_pct']:.2f}%")
    console.print(table)
    console.print(f"Experiment ID: {experiment_id}")
    console.print(f"Validation report: {output_report.resolve()}")


@app.command("replay")
def replay_command(
    cycle_id: int = typer.Option(
        ...,
        "--cycle-id",
        min=1,
        help="Stored strategy cycle id to replay from SQLite snapshots.",
    ),
):
    """Show stored decisions for a previous cycle without calling live APIs."""
    from tradingagents.scheduler.runner import replay_cycle

    try:
        replay = replay_cycle(cycle_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    summary = Table(title=f"Replay Cycle {cycle_id}", box=box.SIMPLE)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Mode", str(replay.get("mode")))
    summary.add_row("Status", str(replay.get("status")))
    summary.add_row("Started", str(replay.get("started_at")))
    summary.add_row("Completed", str(replay.get("completed_at") or "-"))
    summary.add_row("Tickers", ", ".join(replay.get("tickers") or []) or "-")
    if replay.get("error"):
        summary.add_row("Error", str(replay["error"]))
    console.print(summary)

    decisions_table = Table(title="Stored Decisions", box=box.SIMPLE)
    decisions_table.add_column("Ticker", style="cyan")
    decisions_table.add_column("Decision")
    decisions_table.add_column("Reason")
    decisions_table.add_column("Details")
    for decision in replay.get("decisions") or []:
        details = ", ".join(
            f"{k}={v}"
            for k, v in decision.items()
            if k not in {"ticker", "decision", "reason"}
        )
        decisions_table.add_row(
            str(decision.get("ticker", "-")),
            str(decision.get("decision", "-")),
            str(decision.get("reason", "-")),
            details or "-",
        )
    if not replay.get("decisions"):
        decisions_table.add_row("-", "-", "-", "-")
    console.print(decisions_table)


@app.command()
def analyze(
    checkpoint: bool = typer.Option(
        False,
        "--checkpoint",
        help="Enable checkpoint/resume: save state after each node so a crashed run can resume.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
):
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    run_analysis(checkpoint=checkpoint)


if __name__ == "__main__":
    app()
