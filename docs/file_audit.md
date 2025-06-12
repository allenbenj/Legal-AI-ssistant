# File Audit

The following audit documents modules and code flagged during analysis.

## Modules Imported Only by `__main__`

These modules are only referenced by `__main__` according to the dependency graph generated with `pydeps`. They are candidates for consolidation or removal if not required elsewhere.

 -   legal_ai_system.__main__
 -   legal_ai_system.aioredis
 -   legal_ai_system.analytics
 -   legal_ai_system.analytics.quality_classifier
 -   legal_ai_system.config.agent_unified_config
 -   legal_ai_system.config.constants
 -   legal_ai_system.config.workflow_config
 -   legal_ai_system.core.agent_memory_store
 -   legal_ai_system.core.base_agent
 -   legal_ai_system.core.configuration_manager
 -   legal_ai_system.core.embedding_manager
 -   legal_ai_system.core.model_switcher
 -   legal_ai_system.core.models
 -   legal_ai_system.core.performance
 -   legal_ai_system.core.providers
 -   legal_ai_system.core.shared_components
 -   legal_ai_system.core.unified_services
 -   legal_ai_system.core.vector_metadata_repository
 -   legal_ai_system.gui.memory_brain_widget
 -   legal_ai_system.gui.streamlit_app
 -   legal_ai_system.gui.tray_icon
 -   legal_ai_system.langgraph
 -   legal_ai_system.log_setup
 -   legal_ai_system.workflow_engine.models
 -   legal_ai_system.workflows
 -   legal_ai_system.workflows.advanced_langgraph
 -   legal_ai_system.workflows.case_workflow_state
 -   legal_ai_system.workflows.langgraph_setup
 -   legal_ai_system.workflows.lexpredict_pipeline
 -   legal_ai_system.workflows.merge
 -   legal_ai_system.workflows.nodes.legal_error_handling_node
 -   legal_ai_system.workflows.realtime_nodes
 -   legal_ai_system.workflows.retry
 -   legal_ai_system.workflows.routing

## Unused Code Detected by Vulture

 - _RealBaseNode  # unused import (legal_ai_system/agents/agent_nodes.py:20)
 - original_doc_content  # unused variable (legal_ai_system/agents/structural_analysis_agent.py:289)
 - min_similarity  # unused variable (legal_ai_system/core/enhanced_vector_store.py:689)
 - max_depth  # unused variable (legal_ai_system/tests/test_knowledge_graph_reasoning_agent.py:31)
 - relationship_types  # unused variable (legal_ai_system/tests/test_knowledge_graph_reasoning_agent.py:31)
 - _RealStateGraph  # unused import (legal_ai_system/workflows/advanced_langgraph.py:30)
 - _RealEND  # unused import (legal_ai_system/workflows/advanced_langgraph.py:31)
 - _RealStateGraph  # unused import (legal_ai_system/workflows/langgraph_setup.py:34)
 - SystemInitializationError  # unused import (legal_ai_system/services/service_container.py:39)
 - SystemInitializationError  # unused import (legal_ai_system/services/service_container.py:53)
 - SystemInitializationError  # unused import (legal_ai_system/services/service_container.py:67)

## Consolidation or Removal Recommendations

Modules listed above should be reviewed to determine if their functionality can be consolidated into existing packages. Unused imports and variables highlighted by vulture may be removed to simplify maintenance.
