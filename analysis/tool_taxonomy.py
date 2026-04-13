"""Tool category taxonomy for CivBench analysis.

Two classification schemes:

Scheme A: `category` — agent attention allocation (used for stacked area chart).
  Every tool maps to exactly one of: unit_action, state_query, strategic_monitoring,
  diplomacy, game_management, other.

Scheme B: `pmr_subcategory` — Proactive Monitoring Rate subcategories (used for
  Sensorium Effect analysis). Only strategic_monitoring tools are further divided.
  All other tools have pmr_subcategory = None.
"""

from __future__ import annotations

# Scheme A: category

TOOL_CATEGORY: dict[str, str] = {
    # --- unit_action: commands that move or order units ---
    "unit_action": "unit_action",
    "skip_remaining_units": "unit_action",
    "upgrade_unit": "unit_action",
    "city_action": "unit_action",
    "city_attack": "unit_action",
    "spy_action": "unit_action",
    "promote_unit": "unit_action",

    # --- state_query: reading immediate game state (tactical) ---
    "get_game_overview": "state_query",
    "get_units": "state_query",
    "get_cities": "state_query",
    "get_city_production": "state_query",
    "get_map_area": "state_query",
    "get_settle_advisor": "state_query",
    "get_pathing_estimate": "state_query",
    "get_builder_tasks": "state_query",
    "get_unit_promotions": "state_query",
    "get_notifications": "state_query",
    "get_pending_diplomacy": "state_query",
    "get_pending_trades": "state_query",
    "get_trade_destinations": "state_query",
    "get_district_advisor": "state_query",
    "get_wonder_advisor": "state_query",
    "get_purchasable_tiles": "state_query",
    "get_spies": "state_query",
    "get_gp_advisor": "state_query",
    "get_policies": "state_query",
    "get_governors": "state_query",
    "get_pantheon_beliefs": "state_query",
    "get_religion_beliefs": "state_query",
    "get_dedications": "state_query",
    "get_trade_options": "state_query",
    "get_tech_civics": "state_query",

    # --- strategic_monitoring: proactive checks of long-horizon state ---
    "get_victory_progress": "strategic_monitoring",
    "get_religion_spread": "strategic_monitoring",
    "get_strategic_map": "strategic_monitoring",
    "get_global_settle_advisor": "strategic_monitoring",
    "get_trade_routes": "strategic_monitoring",
    "get_empire_resources": "strategic_monitoring",
    "get_great_people": "strategic_monitoring",
    "get_world_congress": "strategic_monitoring",

    # --- diplomacy: proactive diplomatic actions ---
    "get_diplomacy": "diplomacy",
    "send_diplomatic_action": "diplomacy",
    "respond_to_diplomacy": "diplomacy",
    "propose_trade": "diplomacy",
    "respond_to_trade": "diplomacy",
    "propose_peace": "diplomacy",
    "form_alliance": "diplomacy",
    "send_envoy": "diplomacy",
    "get_city_states": "diplomacy",

    # --- other: production, research, economy, religion, governance ---
    "end_turn": "other",
    "set_city_production": "other",
    "set_research": "other",
    "set_policies": "other",
    "set_city_focus": "other",
    "purchase_item": "other",
    "purchase_tile": "other",
    "appoint_governor": "other",
    "assign_governor": "other",
    "promote_governor": "other",
    "change_government": "other",
    "choose_pantheon": "other",
    "found_religion": "other",
    "choose_dedication": "other",
    "recruit_great_person": "other",
    "patronize_great_person": "other",
    "reject_great_person": "other",
    "queue_wc_votes": "other",
    "dismiss_popup": "other",
    "run_lua": "other",
    "vote_world_congress": "other",
    "resolve_city_capture": "other",

    # --- game_management: lifecycle tools (excluded from PMR denominator) ---
    "list_saves": "game_management",
    "load_save": "game_management",
    "load_game_save": "game_management",
    "load_save_from_menu": "game_management",
    "kill_game": "game_management",
    "launch_game": "game_management",
    "restart_and_load": "game_management",
    "get_diary": "game_management",
    "write_diary": "game_management",
    "screenshot": "game_management",
    "quicksave": "game_management",

    # diplomacy: trade testing
    "test_trade": "diplomacy",
}

# Scheme B: pmr_subcategory (subset of strategic_monitoring)

# PMR denominator: all tool calls EXCEPT game_management (and unknown)
# Derived from TOOL_CATEGORY so it stays in sync automatically.
PMR_DENOMINATOR_CATEGORIES = set(TOOL_CATEGORY.values()) - {"game_management"}

# Tools excluded from human-readable tool summaries (used by RAG pipeline).
# These are too noisy or uninformative to surface in planning-vs-execution
# comparison text: pure state queries, lifecycle tools, etc.
SKIP_TOOLS = {
    "get_game_overview", "get_units", "get_map_area", "get_cities",
    "get_city_production", "get_builder_tasks", "get_notifications",
    "get_pending_diplomacy", "get_pending_trades", "get_tech_civics",
    "get_unit_promotions", "get_pathing_estimate", "end_turn",
    "write_diary", "get_diary", "save_game", "screenshot", "run_lua",
    "skip_remaining_units",
}

TOOL_PMR_SUBCATEGORY: dict[str, str | None] = {
    # victory_monitoring: tracks whether agent is winning or losing
    "get_victory_progress": "victory_monitoring",
    "get_religion_spread": "victory_monitoring",

    # strategic_map: spatial awareness beyond immediate area
    "get_strategic_map": "strategic_map",
    "get_global_settle_advisor": "strategic_map",

    # resource_monitoring: economic and great person awareness
    "get_empire_resources": "resource_monitoring",
    "get_great_people": "resource_monitoring",
    "get_world_congress": "resource_monitoring",
    "get_trade_routes": "resource_monitoring",

    # Excluded: get_diplomacy and get_city_states are action precursors as
    # often as pure monitoring — including them would conflate information-
    # gathering with proactive monitoring. Diplomatic state is captured
    # indirectly via the diplomacy category in Scheme A.
}


def categorise(tool_name: str) -> tuple[str, str | None]:
    """Return (category, pmr_subcategory) for a tool name.

    Returns ("unknown", None) for unrecognised tools so new tools
    don't crash the analysis pipeline.
    """
    cat = TOOL_CATEGORY.get(tool_name, "unknown")
    pmr = TOOL_PMR_SUBCATEGORY.get(tool_name, None)
    return cat, pmr


def is_strategic(tool_name: str) -> bool:
    """True if this tool counts toward the strategic_monitoring category."""
    return TOOL_CATEGORY.get(tool_name) == "strategic_monitoring"


def pmr_denominator(tool_name: str) -> bool:
    """True if this tool call should be included in the PMR denominator."""
    return TOOL_CATEGORY.get(tool_name) in PMR_DENOMINATOR_CATEGORIES


ALL_CATEGORIES = {
    "unit_action",
    "state_query",
    "strategic_monitoring",
    "diplomacy",
    "game_management",
    "other",
    "unknown",
}

if __name__ == "__main__":
    # Print summary table for inspection
    from collections import Counter
    cats = Counter(TOOL_CATEGORY.values())
    print("Tool count by category:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<25s} {n}")
    print(f"  {'TOTAL':<25s} {sum(cats.values())}")
    print()
    pmr_tools = {k: v for k, v in TOOL_PMR_SUBCATEGORY.items() if v}
    pmr_cats = Counter(pmr_tools.values())
    print("PMR subcategory breakdown:")
    for sub, n in sorted(pmr_cats.items()):
        tools = [t for t, s in pmr_tools.items() if s == sub]
        print(f"  {sub:<30s} {n}: {', '.join(tools)}")
