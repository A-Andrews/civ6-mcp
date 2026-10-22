#!/usr/bin/env python3
"""
Process agent planning commitments and judge execution against tool logs.

Rubric: Extract CONCRETE, OBSERVABLE commitments only (not vague goals).
- Specific action: build X, research Y, move unit, settle city, call get_victory_progress
- Judge: Y (clearly done), P (partially/different target), N (not done)
"""

import json
import re
from pathlib import Path


def extract_commitments(planning_text):
    """
    Extract concrete commitments from planning text.
    These are specific, observable actions — not vague goals.

    Returns list of {"text": "...", "type": "...", "target": "..."} dicts.
    """
    commitments = []

    if not planning_text or len(planning_text.strip()) < 3:
        return commitments

    # Skip text that is clearly vague goal language
    if re.search(r"Keep pushing|Continue|Keep|maintain|avoid|remember|note|careful", planning_text):
        # Extract only CONCRETE actions, not vague goals
        pass

    # 1. "Build X in City" pattern
    build_in_city_pattern = r"Build\s+(\w+(?:\s+\w+)?)\s+in\s+(\w+)"
    for match in re.finditer(build_in_city_pattern, planning_text, re.IGNORECASE):
        # Check it's not preceded by "need to" or "should"
        before = planning_text[max(0, match.start()-30):match.start()].lower()
        if not any(x in before for x in ["need to", "plan:", "should"]):
            item = match.group(1).strip()
            city = match.group(2).strip()
            # Avoid duplicates with later patterns
            if item.lower() != "campuses":
                commitments.append({
                    "text": f"Build {item} in {city}",
                    "type": "set_city_production",
                    "target": item.lower()
                })

    # 2. "Build more X in City" — also specific action
    build_more_pattern = r"[Bb]uild\s+more\s+(\w+(?:\s+\w+)?)\s+in\s+(\w+)"
    for match in re.finditer(build_more_pattern, planning_text):
        item = match.group(1).strip()
        city = match.group(2).strip()
        # Check if not already added
        text = f"Build {item} in {city}"
        if not any(c["text"] == text for c in commitments):
            commitments.append({
                "text": text,
                "type": "set_city_production",
                "target": item.lower()
            })

    # 3. "research X" or "→ research X" pattern — handle AND/OR chains
    research_blocks = re.finditer(r"research\s+([^.!?;→]*?)(?=\.|!|\?|;|$|→)", planning_text, re.IGNORECASE)
    for block_match in research_blocks:
        block = block_match.group(1).strip()
        before = planning_text[max(0, block_match.start()-50):block_match.start()].lower()
        if any(x in before for x in ["need to", "plan", "should", "may need", "keep", "continue"]):
            continue

        # Split on AND / OR
        techs = re.split(r"\s+(?:AND|OR)\s+", block, flags=re.IGNORECASE)
        for tech_phrase in techs:
            # Extract tech name (first 1-3 words)
            tech = tech_phrase.strip()
            # Remove trailing qualifiers like "(sequential)" or "(need X)"
            tech = re.sub(r"\s*\([^)]*\).*", "", tech).strip()

            if tech and len(tech) > 1 and "eureka" not in tech.lower() and "project" not in tech.lower():
                # Avoid vague phrases like "Research Projects"
                if not re.match(r"^\w+s?\s+(for|of|to)", tech, re.IGNORECASE):
                    commitments.append({
                        "text": f"Research {tech}",
                        "type": "set_research",
                        "target": tech.lower()
                    })

    # 4. "found" or "settler done → found" — indicates settling
    if re.search(r"[Ss]ettler.*done.*found|found.*second.*city|settle(?!\s+for)", planning_text, re.IGNORECASE):
        commitments.append({
            "text": "Settle new city",
            "type": "unit_action_found_city",
            "target": "found"
        })

    # 5. "Spy in X" pattern
    spy_pattern = r"[Ss]py\s+in\s+(\w+)"
    for match in re.finditer(spy_pattern, planning_text):
        location = match.group(1).strip()
        commitments.append({
            "text": f"Spy in {location}",
            "type": "spy_action",
            "target": location.lower()
        })

    # 6. "Propose peace"
    if re.search(r"[Pp]ropose\s+peace", planning_text):
        commitments.append({
            "text": "Propose peace",
            "type": "propose_peace",
            "target": "peace"
        })

    # 7. "send trade routes" or "trade routes to"
    if re.search(r"[Ss]end\s+trade\s+routes?|trade\s+routes?\s+to", planning_text):
        commitments.append({
            "text": "Send trade routes",
            "type": "trade_route",
            "target": "trade_route"
        })

    # 8. "Farm tiles with builders" or "improve tiles"
    if re.search(r"[Ff]arm\s+tiles?|improve.*tiles?", planning_text):
        commitments.append({
            "text": "Farm tiles with builders",
            "type": "unit_action_improve",
            "target": "farm"
        })

    # 9. "Build campuses" or "Build campus" (only if not already in "Build X in City" format)
    if re.search(r"[Bb]uild\s+campuses?(?!\s+in)", planning_text):
        if not any("campus" in c["text"].lower() for c in commitments):
            commitments.append({
                "text": "Build campuses",
                "type": "set_city_production",
                "target": "campus"
            })

    # 10. Check victory progress (if explicit and not vague)
    if re.search(r"[Cc]heck.*victory|victory.*progress|VICTORY BY T\d+", planning_text):
        commitments.append({
            "text": "Check victory progress",
            "type": "get_victory_progress",
            "target": "victory"
        })

    # Deduplicate by (text, type, target)
    seen = set()
    unique = []
    for c in commitments:
        key = (c["text"], c["type"], c["target"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def check_execution(commitment, tools_text):
    """
    Judge whether a commitment was executed in the tools text.

    Returns: ("Y" | "P" | "N", evidence_line)
    """
    if not tools_text:
        return "N", "Tools log empty"

    ctype = commitment["type"]
    target = commitment["target"]
    text = commitment["text"]

    # set_city_production: look for target item in set_city_production call + PRODUCING
    if ctype == "set_city_production":
        # Map item names to expected tool names
        item_mapping = {
            "settler": "UNIT_SETTLER",
            "campus": "DISTRICT_CAMPUS",
            "campuses": "DISTRICT_CAMPUS",
            "trader": "UNIT_TRADER",
            "builder": "UNIT_BUILDER",
            "builders": "UNIT_BUILDER",
            "industrial zone": "DISTRICT_INDUSTRIAL_ZONE",
            "library": "BUILDING_LIBRARY",
            "workshop": "BUILDING_WORKSHOP",
            "mine": "IMPROVEMENT_MINE",
            "factory": "BUILDING_FACTORY",
            "granary": "BUILDING_GRANARY",
            "monument": "BUILDING_MONUMENT",
            "walls": "BUILDING_WALLS",
            "bank": "BUILDING_BANK",
            "university": "BUILDING_UNIVERSITY",
        }

        item_lower = target.lower()
        tool_name = item_mapping.get(item_lower)

        if not tool_name:
            # Try to infer from item name
            tool_name = "UNIT_" + item_lower.upper().replace(" ", "_")

        # Look for set_city_production with this tool name
        if "set_city_production" in tools_text and tool_name in tools_text:
            return "Y", f"set_city_production with {tool_name}"

        # Check if the item appears in PRODUCING output
        if tool_name in tools_text or item_lower in tools_text.lower():
            if "PRODUCING" in tools_text:
                return "Y", f"Item in production"

        return "N", f"{tool_name} not found"

    # set_research: look for target tech + RESEARCHING or PROGRESSING
    if ctype == "set_research":
        tech_lower = target.lower()

        # Map tech names to expected tool names
        tech_mapping = {
            "economics": "TECH_ECONOMICS",
            "buttress": "TECH_BUTTRESS",
            "irrigation": "TECH_IRRIGATION",
            "currency": "TECH_CURRENCY",
            "scientific theory": "TECH_SCIENTIFIC_THEORY",
            "synthetic materials": "TECH_SYNTHETIC_MATERIALS",
            "guidance systems": "TECH_GUIDANCE_SYSTEMS",
            "advanced flight": "TECH_ADVANCED_FLIGHT",
            "lasers": "TECH_LASERS",
            "rocketry": "TECH_ROCKETRY",
            "guilds": "CIVIC_GUILDS",
            "cold war": "CIVIC_COLD_WAR",
            "medieval faires": "CIVIC_MEDIEVAL_FAIRES",
            "early empire": "CIVIC_EARLY_EMPIRE",
            "pottery": "TECH_POTTERY",
            "horseback riding": "TECH_HORSEBACK_RIDING",
            "archery": "TECH_ARCHERY",
            "writing": "TECH_WRITING",
            "code of laws": "CIVIC_CODE_OF_LAWS",
            "engineering": "TECH_ENGINEERING",
            "construction": "TECH_CONSTRUCTION",
            "political philosophy": "CIVIC_POLITICAL_PHILOSOPHY",
            "state workforce": "CIVIC_STATE_WORKFORCE",
        }

        tool_name = tech_mapping.get(tech_lower)
        if not tool_name:
            # Build it
            tech_upper = tech_lower.replace(" ", "_").upper()
            tool_name = "TECH_" + tech_upper  # default to tech

        # Look for set_research or RESEARCHING/PROGRESSING with this tech
        if "set_research" in tools_text and tool_name in tools_text:
            return "Y", f"set_research with {tool_name}"

        if tool_name in tools_text and ("RESEARCHING" in tools_text or "PROGRESSING" in tools_text):
            return "Y", f"{tool_name} in research"

        return "N", f"{tool_name} not found"

    # unit_action_found_city
    if ctype == "unit_action_found_city":
        if "FOUNDED" in tools_text:
            return "Y", "City founded"
        return "N", "No city founded"

    # unit_action_improve (farm/improve tiles)
    if ctype == "unit_action_improve":
        if "IMPROVING" in tools_text or "IMPROVEMENT_FARM" in tools_text:
            return "Y", "Tiles improved"
        return "N", "No improvements"

    # trade_route
    if ctype == "trade_route":
        if "TRADE_ROUTE_STARTED" in tools_text:
            return "Y", "Trade route started"
        return "N", "No trade route"

    # spy_action
    if ctype == "spy_action":
        if "SPY_MISSION" in tools_text:
            return "Y", "Spy mission launched"
        return "N", "No spy action"

    # propose_peace
    if ctype == "propose_peace":
        if "propose_peace" in tools_text.lower() or "Peace established" in tools_text:
            return "Y", "Peace agreed"
        return "N", "No peace"

    # get_victory_progress
    if ctype == "get_victory_progress":
        if "get_victory_progress" in tools_text.lower():
            return "Y", "Victory checked"
        return "N", "No victory check"

    return "N", "Unknown type"


def main():
    input_path = Path("/Users/austinandrews/programming_projects/Machine Learning/civ6-mcp/analysis/rag_inputs_k10.json")
    output_path = Path("/Users/austinandrews/programming_projects/Machine Learning/civ6-mcp/analysis/rag_verdicts_k10.json")

    # Load input
    with open(input_path) as f:
        data = json.load(f)

    verdicts = []
    total_commitments = 0
    y_count = p_count = n_count = 0

    for entry in data:
        planning = entry.get("planning", "")
        tools = entry.get("tools", "")

        # Extract commitments
        commitments = extract_commitments(planning)

        # Judge each commitment
        commitment_verdicts = []
        for commitment in commitments:
            executed, evidence = check_execution(commitment, tools)

            # Truncate text to ~10 words (roughly 60 chars)
            text_short = commitment["text"]
            if len(text_short) > 60:
                text_short = text_short[:57] + "..."

            commitment_verdicts.append({
                "text": text_short,
                "executed": executed,
                "evidence": evidence[:100]
            })

            total_commitments += 1
            if executed == "Y":
                y_count += 1
            elif executed == "P":
                p_count += 1
            else:
                n_count += 1

        verdict_entry = {
            "run_id": entry.get("run_id", ""),
            "model": entry.get("model", ""),
            "scenario": entry.get("scenario", ""),
            "turn": entry.get("turn", 0),
            "commitments": commitment_verdicts
        }
        verdicts.append(verdict_entry)

    # Write output
    with open(output_path, 'w') as f:
        json.dump(verdicts, f, indent=2)

    print(f"Processed {len(data)} turns")
    print(f"Total commitments: {total_commitments}")
    print(f"Y (executed): {y_count}")
    print(f"P (partial): {p_count}")
    print(f"N (not done): {n_count}")
    print(f"\nOutput written to {output_path}")


if __name__ == "__main__":
    main()
