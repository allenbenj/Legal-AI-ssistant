import asyncio
from datetime import datetime, timezone

import pytest

from legal_ai_system.agents.auto_tagging_agent import AutoTaggingAgent


class DummyUMM:
    def __init__(self, stats):
        self.stats = stats

    async def get_tag_learning_stats_async(self, tag_text: str):
        return self.stats.get(tag_text)


class DummyContainer:
    def __init__(self, umm):
        self.umm = umm

    def get_service(self, name: str):
        if name == "unified_memory_manager":
            return self.umm
        return None


@pytest.mark.asyncio
async def test_learning_stats_combines_cache_and_umm():
    iso1 = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
    iso2 = datetime(2024, 1, 2, tzinfo=timezone.utc).isoformat()
    umm_stats = {
        "tag1": {
            "correct_count": 2,
            "incorrect_count": 1,
            "suggested_count": 5,
            "last_updated": iso1,
        },
        "tag2": {
            "correct_count": 3,
            "incorrect_count": 1,
            "suggested_count": 0,
            "last_updated": iso2,
        },
    }
    umm = DummyUMM(umm_stats)
    container = DummyContainer(umm)
    agent = AutoTaggingAgent(container)
    agent.tag_accuracy_scores_cache["tag1"]["correct"] = 1.0
    agent.tag_accuracy_scores_cache["tag1"]["incorrect"] = 1.0
    agent.tag_accuracy_scores_cache["tag1"]["last_updated_ts"] = 10.0
    agent.tag_accuracy_scores_cache["tag3"]["incorrect"] = 2.0
    agent.tag_accuracy_scores_cache["tag3"]["suggested"] = 3.0

    result = await agent.get_learning_statistics_async()

    assert result["umm_status"] == "available"
    assert result["distinct_tags_in_cache"] == 3
    # Overall accuracy uses combined stats
    assert pytest.approx(result["overall_cached_tag_accuracy"], rel=1e-3) == pytest.approx(6 / 11)
    corr_dict = {t: d for t, d in result["top_correct_tags_cache"]}
    assert corr_dict["tag1"]["correct"] == 3.0
    assert corr_dict["tag2"]["correct"] == 3.0
