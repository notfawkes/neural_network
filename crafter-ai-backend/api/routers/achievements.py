"""Achievements router."""
from fastapi import APIRouter
from typing import Dict, Any, List

from core.config import settings
from services.metrics_service import MetricsService

router = APIRouter(prefix="/api/achievements", tags=["achievements"])
metrics_service = MetricsService(settings.workspace_dir)

ACHIEVEMENT_METADATA = [
    {"name": "collect_coal", "title": "Collect Coal", "category": "mining", "icon": "🪨"},
    {"name": "collect_diamond", "title": "Collect Diamond", "category": "mining", "icon": "💎"},
    {"name": "collect_drink", "title": "Collect Drink", "category": "survival", "icon": "💧"},
    {"name": "collect_iron", "title": "Collect Iron", "category": "mining", "icon": "⛓️"},
    {"name": "collect_sapling", "title": "Collect Sapling", "category": "farming", "icon": "🌱"},
    {"name": "collect_stone", "title": "Collect Stone", "category": "mining", "icon": "🗿"},
    {"name": "collect_wood", "title": "Collect Wood", "category": "foraging", "icon": "🪵"},
    {"name": "defeat_skeleton", "title": "Defeat Skeleton", "category": "combat", "icon": "🏹"},
    {"name": "defeat_zombie", "title": "Defeat Zombie", "category": "combat", "icon": "🧟"},
    {"name": "eat_cow", "title": "Eat Cow", "category": "survival", "icon": "🥩"},
    {"name": "eat_plant", "title": "Eat Plant", "category": "survival", "icon": "🍎"},
    {"name": "make_iron_pickaxe", "title": "Make Iron Pickaxe", "category": "crafting", "icon": "⛏️"},
    {"name": "make_iron_sword", "title": "Make Iron Sword", "category": "crafting", "icon": "⚔️"},
    {"name": "make_stone_pickaxe", "title": "Make Stone Pickaxe", "category": "crafting", "icon": "⛏️"},
    {"name": "make_stone_sword", "title": "Make Stone Sword", "category": "crafting", "icon": "🗡️"},
    {"name": "make_wood_pickaxe", "title": "Make Wood Pickaxe", "category": "crafting", "icon": "⛏️"},
    {"name": "make_wood_sword", "title": "Make Wood Sword", "category": "crafting", "icon": "🗡️"},
    {"name": "place_furnace", "title": "Place Furnace", "category": "building", "icon": "🔥"},
    {"name": "place_plant", "title": "Place Plant", "category": "farming", "icon": "🌿"},
    {"name": "place_stone", "title": "Place Stone", "category": "building", "icon": "🧱"},
    {"name": "place_table", "title": "Place Crafting Table", "category": "building", "icon": "🪵"},
    {"name": "wake_up", "title": "Wake Up", "category": "survival", "icon": "☀️"},
]


@router.get("")
async def get_achievements():
    metrics = metrics_service.get_full_metrics()
    rates = metrics.get("achievement_rates", {})

    achievements = []
    for meta in ACHIEVEMENT_METADATA:
        name = meta["name"]
        rate = rates.get(name, 0.0)
        achievements.append({
            **meta,
            "success_rate": rate,
            "unlocked": rate > 0.0,
        })

    return {
        "achievements": achievements,
        "crafter_score": metrics.get("crafter_score", 0.0),
        "total_episodes": metrics.get("total_episodes", 0),
    }
