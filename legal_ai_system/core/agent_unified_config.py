"""
Unified Agent Configuration
Ensures all agents use Grok-Mini for LLM operations and shared memory for storage
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from legal_ai_system.config.agent_grok_config import (
        create_agent_grok_config, get_agent_grok_manager, AGENT_DEFAULT_MODEL
    )
    from legal_ai_system.memory.unified_memory_manager import UnifiedMemoryManager, MemoryType
    from legal_ai_system.services.service_container import ServiceContainer
except ImportError:
    # Fallback for relative imports
    try:
        from .agent_grok_config import (
            create_agent_grok_config, get_agent_grok_manager, AGENT_DEFAULT_MODEL
        )
        from ..memory.unified_memory_manager import UnifiedMemoryManager, MemoryType
        from ..services.service_container import ServiceContainer
    except ImportError:
        # Final fallback - create minimal classes
        class ServiceContainer:
            def __init__(self):
                self.services = {}
            def register_service(self, name, service):
                self.services[name] = service
            def get_service(self, name):
                return self.services.get(name)
        class UnifiedMemoryManager:
            def __init__(self, **kwargs):
                pass
        class MemoryType:
            AGENT_SPECIFIC = "agent_specific"
            SESSION_KNOWLEDGE = "session_knowledge"
        # Mock functions
        def create_agent_grok_config(**kwargs): return {}
        def get_agent_grok_manager(**kwargs): return None
        AGENT_DEFAULT_MODEL = "grok-3-mini"

async def configure_all_agents_unified(
    service_container: ServiceContainer,
    xai_api_key: Optional[str] = None,
    memory_db_path: str = "./storage/databases/unified_memory.db"
) -> Dict[str, Any]:
    """
    Configure all agents to use Grok-Mini and shared memory
    
    Args:
        service_container: The service container instance
        xai_api_key: XAI API key for Grok models
        memory_db_path: Path to the unified memory database
    
    Returns:
        Configuration summary
    """
    
    print("🔧 Configuring all agents for unified operation...")
    
    config_summary = {
        "llm_configured": False,
        "memory_configured": False,
        "agents_updated": 0,
        "services_registered": [],
        "errors": []
    }
    
    try:
        # 1. Configure LLM (Grok-Mini) for all agents
        print(f"📡 Configuring LLM: {AGENT_DEFAULT_MODEL}")
        
        api_key = xai_api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI API key required. Set XAI_API_KEY environment variable or provide xai_api_key parameter.")
        
        # Create and register Grok configuration
        agent_llm_config = create_agent_grok_config(api_key)
        await service_container.register_service('llm_config', agent_llm_config)
        await service_container.register_service('agent_llm_config', agent_llm_config)
        
        # Register Grok manager
        grok_manager = get_agent_grok_manager(api_key)
        await service_container.register_service('grok_manager', grok_manager)
        
        config_summary["llm_configured"] = True
        config_summary["services_registered"].append("llm_config")
        config_summary["services_registered"].append("grok_manager")
        print(f"✅ LLM configured: {AGENT_DEFAULT_MODEL}")
        
    except Exception as e:
        error_msg = f"Failed to configure LLM: {e}"
        config_summary["errors"].append(error_msg)
        print(f"❌ {error_msg}")
    
    try:
        # 2. Configure Unified Memory Manager
        print("🧠 Configuring shared memory...")
        
        # Create and initialize memory manager
        memory_manager = UnifiedMemoryManager(
            db_path_str=memory_db_path,
            service_config={
                "enable_agent_memory": True,
                "enable_session_memory": True,
                "enable_context_memory": True,
                "max_memory_entries": 10000,
                "memory_cleanup_interval": 3600  # 1 hour
            }
        )
        
        # Register memory manager in service container
        await service_container.register_service('unified_memory_manager', memory_manager)
        await service_container.register_service('memory_manager', memory_manager)  # Alias
        
        config_summary["memory_configured"] = True
        config_summary["services_registered"].append("unified_memory_manager")
        print("✅ Shared memory configured")
        
    except Exception as e:
        error_msg = f"Failed to configure memory: {e}"
        config_summary["errors"].append(error_msg)
        print(f"❌ {error_msg}")
    
    try:
        # 3. Register agent helper services
        print("🔧 Registering agent helper services...")
        
        # Register agent configuration helper
        agent_config_helper = AgentConfigHelper(service_container)
        await service_container.register_service('agent_config_helper', agent_config_helper)
        
        config_summary["services_registered"].append("agent_config_helper")
        print("✅ Agent helper services registered")
        
    except Exception as e:
        error_msg = f"Failed to register helper services: {e}"
        config_summary["errors"].append(error_msg)
        print(f"❌ {error_msg}")
    
    # 4. Summary
    success = config_summary["llm_configured"] and config_summary["memory_configured"]
    
    if success:
        print("\n🎉 ALL AGENTS CONFIGURED SUCCESSFULLY!")
        print(f"📊 LLM Model: {AGENT_DEFAULT_MODEL}")
        print(f"🧠 Memory: Unified Memory Manager")
        print(f"📁 Memory DB: {memory_db_path}")
        print(f"🔧 Services: {len(config_summary['services_registered'])} registered")
    else:
        print(f"\n❌ Configuration incomplete. Errors: {len(config_summary['errors'])}")
        for error in config_summary["errors"]:
            print(f"   • {error}")
    
    return config_summary

class AgentConfigHelper:
    """Helper class to assist agents with configuration and service access"""
    
    def __init__(self, service_container: ServiceContainer):
        self.service_container = service_container
    
    def get_llm_config_for_agent(self, agent_name: str) -> Dict[str, Any]:
        """Get LLM configuration optimized for specific agent"""
        try:
            grok_manager = self.service_container.get_service('grok_manager')
            if grok_manager:
                return grok_manager.get_agent_config(agent_name)
        except Exception as e:
            print(f"Warning: Could not get agent-specific LLM config: {e}")
        
        # Fallback to default
        return {
            "llm_provider": "xai",
            "llm_model": AGENT_DEFAULT_MODEL,
            "llm_temperature": 0.3,
            "llm_max_tokens": 4096
        }
    
    def get_memory_manager_for_agent(self, agent_name: str):
        """Get memory manager instance for agent"""
        return self.service_container.get_service('unified_memory_manager')
    
    def store_agent_memory(self, agent_name: str, key: str, value: Any, 
                          session_id: Optional[str] = None, memory_type: MemoryType = MemoryType.AGENT_SPECIFIC):
        """Store data in agent memory"""
        memory_manager = self.get_memory_manager_for_agent(agent_name)
        if memory_manager:
            return memory_manager.store_agent_memory(
                agent_name=agent_name,
                key=key,
                value=value,
                session_id=session_id,
                memory_type=memory_type
            )
        return False
    
    def retrieve_agent_memory(self, agent_name: str, key: str, 
                             session_id: Optional[str] = None, memory_type: MemoryType = MemoryType.AGENT_SPECIFIC):
        """Retrieve data from agent memory"""
        memory_manager = self.get_memory_manager_for_agent(agent_name)
        if memory_manager:
            return memory_manager.retrieve_agent_memory(
                agent_name=agent_name,
                key=key,
                session_id=session_id,
                memory_type=memory_type
            )
        return None

def create_agent_memory_mixin():
    """Create a mixin class to add memory functionality to agents"""
    
    class AgentMemoryMixin:
        """Mixin to add unified memory capabilities to agents"""
        
        def get_memory_manager(self):
            """Get the unified memory manager"""
            if hasattr(self, 'service_container') and self.service_container:
                return self.service_container.get_service('unified_memory_manager')
            return None
        
        def get_agent_config_helper(self):
            """Get the agent configuration helper"""
            if hasattr(self, 'service_container') and self.service_container:
                return self.service_container.get_service('agent_config_helper')
            return None
        
        def store_memory(self, key: str, value: Any, session_id: Optional[str] = None, 
                        memory_type: MemoryType = MemoryType.AGENT_SPECIFIC):
            """Store data in memory with agent-specific namespace"""
            helper = self.get_agent_config_helper()
            if helper:
                return helper.store_agent_memory(
                    agent_name=getattr(self, 'name', self.__class__.__name__),
                    key=key,
                    value=value,
                    session_id=session_id,
                    memory_type=memory_type
                )
            return False
        
        def retrieve_memory(self, key: str, session_id: Optional[str] = None,
                           memory_type: MemoryType = MemoryType.AGENT_SPECIFIC):
            """Retrieve data from memory with agent-specific namespace"""
            helper = self.get_agent_config_helper()
            if helper:
                return helper.retrieve_agent_memory(
                    agent_name=getattr(self, 'name', self.__class__.__name__),
                    key=key,
                    session_id=session_id,
                    memory_type=memory_type
                )
            return None
        
        def get_optimized_llm_config(self):
            """Get LLM configuration optimized for this agent"""
            helper = self.get_agent_config_helper()
            if helper:
                return helper.get_llm_config_for_agent(
                    getattr(self, 'name', self.__class__.__name__)
                )
            return {}
        
        def store_analysis_result(self, result: Any, document_id: str, session_id: Optional[str] = None):
            """Store analysis result in memory for future reference"""
            return self.store_memory(
                key=f"analysis_result_{document_id}",
                value=result,
                session_id=session_id,
                memory_type=MemoryType.SESSION_KNOWLEDGE
            )
        
        def retrieve_previous_analysis(self, document_id: str, session_id: Optional[str] = None):
            """Retrieve previous analysis result if available"""
            return self.retrieve_memory(
                key=f"analysis_result_{document_id}",
                session_id=session_id,
                memory_type=MemoryType.SESSION_KNOWLEDGE
            )
        
        def store_entities(self, entities: List[Dict], document_id: str, session_id: Optional[str] = None):
            """Store extracted entities in session knowledge"""
            return self.store_memory(
                key=f"entities_{document_id}",
                value=entities,
                session_id=session_id,
                memory_type=MemoryType.SESSION_KNOWLEDGE
            )
        
        def retrieve_entities(self, document_id: str, session_id: Optional[str] = None):
            """Retrieve previously extracted entities"""
            return self.retrieve_memory(
                key=f"entities_{document_id}",
                session_id=session_id,
                memory_type=MemoryType.SESSION_KNOWLEDGE
            )
    
    return AgentMemoryMixin

def get_agent_configuration_status(service_container: ServiceContainer) -> Dict[str, Any]:
    """Get current agent configuration status"""
    
    status = {
        "llm_configured": False,
        "memory_configured": False,
        "services_available": [],
        "missing_services": [],
        "ready": False
    }
    
    # Check required services
    required_services = [
        'llm_config',
        'unified_memory_manager',
        'grok_manager',
        'agent_config_helper'
    ]
    
    for service_name in required_services:
        try:
            service = service_container.get_service(service_name)
            if service:
                status["services_available"].append(service_name)
            else:
                status["missing_services"].append(service_name)
        except Exception:
            status["missing_services"].append(service_name)
    
    # Determine configuration status
    status["llm_configured"] = 'llm_config' in status["services_available"]
    status["memory_configured"] = 'unified_memory_manager' in status["services_available"]
    status["ready"] = len(status["missing_services"]) == 0
    
    return status

def validate_agent_setup(service_container: ServiceContainer) -> bool:
    """Validate that agents are properly configured"""
    
    print("\n🔍 Validating agent configuration...")
    
    status = get_agent_configuration_status(service_container)
    
    print(f"LLM Configuration: {'✅' if status['llm_configured'] else '❌'}")
    print(f"Memory Configuration: {'✅' if status['memory_configured'] else '❌'}")
    print(f"Services Available: {len(status['services_available'])}/{len(status['services_available']) + len(status['missing_services'])}")
    
    if status["missing_services"]:
        print(f"Missing Services: {status['missing_services']}")
    
    if status["ready"]:
        print("🎉 Agent configuration is complete!")
        
        # Test memory operation
        try:
            memory_manager = service_container.get_service('unified_memory_manager')
            test_result = memory_manager.store_agent_memory(
                agent_name="test_agent",
                key="config_test",
                value={"test": True},
                memory_type=MemoryType.AGENT_SPECIFIC
            )
            if test_result:
                print("✅ Memory operation test successful")
            else:
                print("⚠️ Memory operation test failed")
        except Exception as e:
            print(f"⚠️ Memory test error: {e}")
        
        # Test LLM configuration
        try:
            grok_manager = service_container.get_service('grok_manager')
            validation = grok_manager.validate_agent_grok_setup()
            if validation["ready"]:
                print("✅ LLM configuration test successful")
            else:
                print("⚠️ LLM configuration test failed")
        except Exception as e:
            print(f"⚠️ LLM test error: {e}")
    else:
        print("❌ Agent configuration incomplete")
    
    return status["ready"]

# Example usage function
def setup_agents_example(xai_api_key: str):
    """Example of how to set up agents with unified configuration"""
    
    print("Setting up Legal AI System agents...")
    
    # 1. Create service container
    service_container = ServiceContainer()
    
    # 2. Configure all agents
    config_result = configure_all_agents_unified(
        service_container=service_container,
        xai_api_key=xai_api_key
    )
    
    # 3. Validate setup
    is_ready = validate_agent_setup(service_container)
    
    if is_ready:
        print("\n🎯 All agents are now configured to use:")
        print(f"   • LLM: {AGENT_DEFAULT_MODEL} (Grok-Mini)")
        print("   • Memory: Unified Memory Manager")
        print("   • Storage: SQLite database")
        print("   • Features: Agent-specific configs, session memory, context management")
        
        return service_container
    else:
        print("\n❌ Setup failed. Please check configuration.")
        return None

if __name__ == "__main__":
    print("Agent Unified Configuration")
    print("=" * 50)
    print("This module configures all agents to use:")
    print(f"• LLM: {AGENT_DEFAULT_MODEL}")
    print("• Memory: Unified Memory Manager")
    print("• Storage: Shared SQLite database")
    print("\nTo use:")
    print("from config.agent_unified_config import configure_all_agents_unified")
    print("configure_all_agents_unified(service_container, xai_api_key)")