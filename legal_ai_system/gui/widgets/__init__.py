"""Widget subpackage providing reusable PyQt6 components."""

from .legal_ai_charts import (
    AnalyticsDashboardWidget,
    PieChartWidget,
    BarChartWidget,
)
from .agent_manager_widget import AgentManagerWidget

__all__ = [
    "AnalyticsDashboardWidget",
    "PieChartWidget",
    "BarChartWidget",
    "AgentManagerWidget",
]

