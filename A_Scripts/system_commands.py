"""
CLI Commands for Real-Time Legal AI System Management.

This module provides command-line interface commands for managing
graph synchronization, vector optimization, and system operations.
"""

import asyncio
import click
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import time

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.services import ServiceContainer
from workflows.realtime_analysis_workflow import RealTimeAnalysisWorkflow


class LegalAICLI:
    """Command-line interface for the Legal AI System."""
    
    def __init__(self):
        self.services: Optional[ServiceContainer] = None
        self.workflow: Optional[RealTimeAnalysisWorkflow] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the CLI system."""
        if self.initialized:
            return
        
        try:
            click.echo("🚀 Initializing Legal AI System...")
            
            # Initialize services
            from core.services import initialize_services
            self.services = await initialize_services()
            
            # Initialize workflow
            self.workflow = RealTimeAnalysisWorkflow(self.services)
            await self.workflow.initialize()
            
            self.initialized = True
            click.echo("✅ Legal AI System initialized successfully")
            
        except Exception as e:
            click.echo(f"❌ Initialization failed: {e}", err=True)
            raise
    
    async def close(self):
        """Close the CLI system."""
        if self.workflow:
            await self.workflow.close()
        if self.services:
            await self.services.close()
        self.initialized = False


# Global CLI instance
cli_instance = LegalAICLI()


@click.group()
@click.version_option(version="1.0.0", prog_name="Legal AI System CLI")
def legal_ai():
    """Legal AI System - Real-time document analysis and knowledge management."""
    pass


@legal_ai.command()
@click.option('--force', is_flag=True, help='Force full synchronization')
@click.option('--timeout', default=300, help='Timeout in seconds')
def sync_graph(force: bool, timeout: int):
    """Synchronize entities with the Neo4j Knowledge Graph."""
    async def _sync_graph():
        await cli_instance.initialize()
        
        click.echo("🔄 Starting knowledge graph synchronization...")
        start_time = time.time()
        
        try:
            if force:
                result = await cli_instance.workflow.graph_manager.force_full_sync()
                click.echo("📊 Full synchronization completed")
            else:
                # Get current sync status
                stats = await cli_instance.workflow.graph_manager.get_realtime_stats()
                click.echo(f"📈 Current status: {stats['pending_sync']} items pending sync")
                
                if stats['pending_sync'] > 0:
                    # Wait for background sync or force if timeout
                    waited = 0
                    while stats['pending_sync'] > 0 and waited < timeout:
                        await asyncio.sleep(5)
                        waited += 5
                        stats = await cli_instance.workflow.graph_manager.get_realtime_stats()
                        if waited % 30 == 0:  # Update every 30 seconds
                            click.echo(f"⏳ Waiting... {stats['pending_sync']} items remaining")
                    
                    if stats['pending_sync'] > 0:
                        click.echo("⚠️  Timeout reached, forcing synchronization...")
                        result = await cli_instance.workflow.graph_manager.force_full_sync()
                    else:
                        result = {'sync_completed': True, 'message': 'Background sync completed'}
                else:
                    result = {'sync_completed': True, 'message': 'Already synchronized'}
            
            elapsed = time.time() - start_time
            
            if result.get('sync_completed'):
                click.echo(f"✅ Graph synchronization completed in {elapsed:.2f}s")
                click.echo(f"📊 Items synced: {result.get('items_synced', 'N/A')}")
                if result.get('errors', 0) > 0:
                    click.echo(f"⚠️  Errors encountered: {result['errors']}")
            else:
                click.echo(f"❌ Synchronization failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            click.echo(f"❌ Synchronization failed: {e}", err=True)
    
    asyncio.run(_sync_graph())


@legal_ai.command()
@click.argument('documents', nargs=-1, type=click.Path(exists=True))
@click.option('--directory', type=click.Path(exists=True), help='Process all documents in directory')
@click.option('--pattern', default='*.pdf', help='File pattern for directory processing')
@click.option('--parallel', is_flag=True, help='Process documents in parallel')
def build_knowledge_graph(documents, directory, pattern, parallel):
    """Automatically build the Knowledge Graph from entities in documents."""
    async def _build_graph():
        await cli_instance.initialize()
        
        # Collect document paths
        doc_paths = list(documents) if documents else []
        
        if directory:
            from pathlib import Path
            dir_path = Path(directory)
            doc_paths.extend([str(p) for p in dir_path.glob(pattern)])
        
        if not doc_paths:
            click.echo("❌ No documents specified. Use --directory or provide document paths.")
            return
        
        click.echo(f"🏗️  Building knowledge graph from {len(doc_paths)} documents...")
        
        total_entities = 0
        total_relationships = 0
        successful_docs = 0
        
        start_time = time.time()
        
        try:
            if parallel:
                # Process documents in parallel
                tasks = []
                for doc_path in doc_paths:
                    task = cli_instance.workflow.process_document_realtime(doc_path)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        click.echo(f"❌ Failed to process {doc_paths[i]}: {result}")
                    else:
                        successful_docs += 1
                        total_entities += result.graph_updates.get('nodes_created', 0)
                        total_relationships += result.graph_updates.get('edges_created', 0)
                        click.echo(f"✅ Processed: {Path(doc_paths[i]).name}")
            else:
                # Process documents sequentially
                for i, doc_path in enumerate(doc_paths, 1):
                    try:
                        click.echo(f"📄 Processing ({i}/{len(doc_paths)}): {Path(doc_path).name}")
                        
                        result = await cli_instance.workflow.process_document_realtime(doc_path)
                        
                        successful_docs += 1
                        total_entities += result.graph_updates.get('nodes_created', 0)
                        total_relationships += result.graph_updates.get('edges_created', 0)
                        
                        click.echo(f"   ✅ Entities: {result.graph_updates.get('nodes_created', 0)}, "
                                 f"Relations: {result.graph_updates.get('edges_created', 0)}")
                        
                    except Exception as e:
                        click.echo(f"   ❌ Failed: {e}")
            
            elapsed = time.time() - start_time
            
            click.echo(f"\n🎉 Knowledge graph building completed in {elapsed:.2f}s")
            click.echo(f"📊 Results:")
            click.echo(f"   • Documents processed: {successful_docs}/{len(doc_paths)}")
            click.echo(f"   • Total entities created: {total_entities}")
            click.echo(f"   • Total relationships created: {total_relationships}")
            
            # Force final synchronization
            click.echo(f"\n🔄 Performing final synchronization...")
            sync_result = await cli_instance.workflow.force_system_sync()
            click.echo(f"✅ Final sync completed")
        
        except Exception as e:
            click.echo(f"❌ Knowledge graph building failed: {e}", err=True)
    
    asyncio.run(_build_graph())


@legal_ai.command()
@click.option('--force-rebuild', is_flag=True, help='Force rebuild of FAISS index')
@click.option('--clean-cache', is_flag=True, help='Clean query cache')
@click.option('--update-frequent', is_flag=True, help='Update frequent entities cache')
def optimize_vector_store(force_rebuild, clean_cache, update_frequent):
    """Trigger optimization of the Vector Store for faster lookups."""
    async def _optimize_store():
        await cli_instance.initialize()
        
        click.echo("⚡ Optimizing vector store...")
        start_time = time.time()
        
        try:
            # Get current stats
            stats_before = await cli_instance.workflow.vector_store.get_performance_stats()
            click.echo(f"📊 Current state:")
            click.echo(f"   • Total vectors: {stats_before['total_vectors']}")
            click.echo(f"   • Cache hit rate: {stats_before['cache_hit_rate']:.2%}")
            click.echo(f"   • Avg search time: {stats_before['avg_search_time']:.3f}s")
            
            # Perform optimization
            optimization_params = {
                'force_rebuild_index': force_rebuild,
                'clean_cache': clean_cache,
                'update_frequent_entities': update_frequent
            }
            
            result = await cli_instance.workflow.vector_store.optimize_performance()
            
            elapsed = time.time() - start_time
            
            if result.get('optimization_completed'):
                click.echo(f"✅ Vector store optimization completed in {elapsed:.2f}s")
                click.echo(f"📊 Optimization results:")
                click.echo(f"   • Vectors processed: {result.get('total_vectors', 'N/A')}")
                click.echo(f"   • Frequent entities cached: {result.get('frequent_vectors_cached', 'N/A')}")
                click.echo(f"   • Cache entries: {result.get('cache_entries', 'N/A')}")
                
                # Get updated stats
                stats_after = await cli_instance.workflow.vector_store.get_performance_stats()
                
                if stats_after['avg_search_time'] < stats_before['avg_search_time']:
                    improvement = ((stats_before['avg_search_time'] - stats_after['avg_search_time']) / 
                                 stats_before['avg_search_time']) * 100
                    click.echo(f"🚀 Search performance improved by {improvement:.1f}%")
            else:
                click.echo(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            click.echo(f"❌ Vector store optimization failed: {e}", err=True)
    
    asyncio.run(_optimize_store())


@legal_ai.command()
@click.option('--format', 'output_format', type=click.Choice(['json', 'table']), default='table',
              help='Output format')
def system_status(output_format):
    """Display comprehensive system status and statistics."""
    async def _get_status():
        await cli_instance.initialize()
        
        try:
            stats = await cli_instance.workflow.get_system_stats()
            
            if output_format == 'json':
                click.echo(json.dumps(stats, indent=2, default=str))
            else:
                # Table format
                click.echo("🔍 Legal AI System Status")
                click.echo("=" * 50)
                
                # Workflow stats
                workflow_stats = stats.get('workflow_stats', {})
                click.echo(f"\n📊 Workflow Statistics:")
                click.echo(f"   Documents processed: {workflow_stats.get('documents_processed', 0)}")
                click.echo(f"   Avg processing time: {workflow_stats.get('avg_processing_time', 0):.2f}s")
                click.echo(f"   Min/Max time: {workflow_stats.get('min_processing_time', 0):.2f}s / {workflow_stats.get('max_processing_time', 0):.2f}s")
                
                # Graph stats
                graph_stats = stats.get('graph_stats', {})
                click.echo(f"\n🕸️  Knowledge Graph:")
                click.echo(f"   Total nodes: {graph_stats.get('total_nodes', 0)}")
                click.echo(f"   Total edges: {graph_stats.get('total_edges', 0)}")
                click.echo(f"   Pending sync: {graph_stats.get('pending_sync', 0)}")
                click.echo(f"   Neo4j available: {'✅' if graph_stats.get('neo4j_available') else '❌'}")
                
                # Vector stats
                vector_stats = stats.get('vector_stats', {})
                click.echo(f"\n🔍 Vector Store:")
                click.echo(f"   Total vectors: {vector_stats.get('total_vectors', 0)}")
                click.echo(f"   Cache hit rate: {vector_stats.get('cache_hit_rate', 0):.2%}")
                click.echo(f"   Avg search time: {vector_stats.get('avg_search_time', 0):.3f}s")
                click.echo(f"   FAISS available: {'✅' if vector_stats.get('faiss_available') else '❌'}")
                
                # Memory stats
                memory_stats = stats.get('memory_stats', {})
                click.echo(f"\n🧠 Review Memory:")
                click.echo(f"   Pending reviews: {memory_stats.get('pending_reviews', 0)}")
                click.echo(f"   Auto-approval enabled: {'✅' if memory_stats.get('auto_approval_enabled') else '❌'}")
                
                # Extraction stats
                extraction_stats = stats.get('extraction_stats', {})
                models_available = extraction_stats.get('models_available', {})
                click.echo(f"\n🔬 Extraction Models:")
                click.echo(f"   spaCy: {'✅' if models_available.get('spacy') else '❌'}")
                click.echo(f"   Blackstone: {'✅' if models_available.get('blackstone') else '❌'}")
                click.echo(f"   Flair: {'✅' if models_available.get('flair') else '❌'}")
                click.echo(f"   LLM: {'✅' if models_available.get('llm') else '❌'}")
                
                # System health
                health = stats.get('system_health', {})
                click.echo(f"\n💚 System Health:")
                click.echo(f"   Components initialized: {'✅' if health.get('components_initialized') else '❌'}")
                click.echo(f"   Real-time sync: {'✅' if health.get('real_time_sync_enabled') else '❌'}")
        
        except Exception as e:
            click.echo(f"❌ Failed to get system status: {e}", err=True)
    
    asyncio.run(_get_status())


@legal_ai.command()
@click.argument('document_path', type=click.Path(exists=True))
@click.option('--verbose', is_flag=True, help='Show detailed extraction results')
@click.option('--targeted', is_flag=True, help='Include targeted extractions')
def analyze_document(document_path, verbose, targeted):
    """Analyze a single document and display extraction results."""
    async def _analyze():
        await cli_instance.initialize()
        
        click.echo(f"📄 Analyzing document: {Path(document_path).name}")
        start_time = time.time()
        
        try:
            # Set progress callback
            def progress_callback(message, progress):
                click.echo(f"   {message} ({progress:.0%})")
            
            cli_instance.workflow.register_progress_callback(progress_callback)
            
            # Process document
            result = await cli_instance.workflow.process_document_realtime(
                document_path, enable_targeted=targeted
            )
            
            elapsed = time.time() - start_time
            
            click.echo(f"\n✅ Analysis completed in {elapsed:.2f}s")
            click.echo(f"📊 Results Summary:")
            click.echo(f"   • Overall confidence: {result.confidence_scores.get('overall', 0):.2%}")
            click.echo(f"   • Entities extracted: {len(result.hybrid_extraction.validated_entities)}")
            click.echo(f"   • Relationships found: {len(result.ontology_extraction.relationships)}")
            click.echo(f"   • Graph nodes created: {result.graph_updates.get('nodes_created', 0)}")
            click.echo(f"   • Vectors added: {result.vector_updates.get('vectors_added', 0)}")
            
            if verbose:
                click.echo(f"\n🔍 Detailed Results:")
                
                # Show entities
                click.echo(f"\n   Entities ({len(result.hybrid_extraction.validated_entities)}):")
                for entity in result.hybrid_extraction.validated_entities[:10]:  # Limit display
                    click.echo(f"     • {entity.entity_text} ({entity.consensus_type}) - {entity.confidence:.2f}")
                
                # Show targeted extractions
                if targeted and result.hybrid_extraction.targeted_extractions:
                    click.echo(f"\n   Targeted Extractions:")
                    for extraction_type, extractions in result.hybrid_extraction.targeted_extractions.items():
                        click.echo(f"     {extraction_type}: {len(extractions)} items")
                        for item in extractions[:3]:  # Show first 3
                            description = item.get('description', item.get('type', 'N/A'))[:100]
                            click.echo(f"       - {description}...")
            
        except Exception as e:
            click.echo(f"❌ Document analysis failed: {e}", err=True)
    
    asyncio.run(_analyze())


@legal_ai.command()
@click.option('--threshold', type=float, help='New confidence threshold')
@click.option('--auto-approve', type=float, help='Auto-approve threshold')
@click.option('--review', type=float, help='Review threshold')
@click.option('--reject', type=float, help='Reject threshold')
def configure_thresholds(threshold, auto_approve, review, reject):
    """Configure confidence thresholds for processing and review."""
    async def _configure():
        await cli_instance.initialize()
        
        click.echo("⚙️  Configuring system thresholds...")
        
        try:
            updates = {}
            
            if threshold is not None:
                cli_instance.workflow.confidence_threshold = threshold
                updates['confidence_threshold'] = threshold
            
            if any([auto_approve, review, reject]):
                threshold_updates = {}
                if auto_approve is not None:
                    threshold_updates['auto_approve_threshold'] = auto_approve
                if review is not None:
                    threshold_updates['review_threshold'] = review
                if reject is not None:
                    threshold_updates['reject_threshold'] = reject
                
                await cli_instance.workflow.reviewable_memory.update_thresholds(threshold_updates)
                updates.update(threshold_updates)
            
            if updates:
                click.echo("✅ Thresholds updated:")
                for key, value in updates.items():
                    click.echo(f"   • {key}: {value}")
            else:
                click.echo("ℹ️  No threshold changes specified")
                
                # Show current thresholds
                stats = await cli_instance.workflow.reviewable_memory.get_review_stats()
                current_thresholds = stats.get('thresholds', {})
                click.echo("📊 Current thresholds:")
                click.echo(f"   • Confidence threshold: {cli_instance.workflow.confidence_threshold}")
                for key, value in current_thresholds.items():
                    click.echo(f"   • {key}: {value}")
        
        except Exception as e:
            click.echo(f"❌ Configuration failed: {e}", err=True)
    
    asyncio.run(_configure())


@legal_ai.command()
def force_system_sync():
    """Force synchronization across all system components."""
    async def _force_sync():
        await cli_instance.initialize()
        
        click.echo("🔄 Forcing system-wide synchronization...")
        start_time = time.time()
        
        try:
            result = await cli_instance.workflow.force_system_sync()
            elapsed = time.time() - start_time
            
            click.echo(f"✅ System synchronization completed in {elapsed:.2f}s")
            click.echo(f"📊 Sync results:")
            
            for component, status in result.items():
                if component == 'error':
                    continue
                
                if isinstance(status, dict):
                    if status.get('sync_completed') or status.get('optimization_completed'):
                        click.echo(f"   • {component}: ✅ Success")
                    else:
                        click.echo(f"   • {component}: ❌ Failed")
                else:
                    click.echo(f"   • {component}: {status}")
            
            if 'error' in result:
                click.echo(f"⚠️  Errors encountered: {result['error']}")
        
        except Exception as e:
            click.echo(f"❌ System sync failed: {e}", err=True)
    
    asyncio.run(_force_sync())


if __name__ == '__main__':
    # Ensure CLI instance is properly closed
    import atexit
    
    def cleanup():
        if cli_instance.initialized:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(cli_instance.close())
                else:
                    loop.run_until_complete(cli_instance.close())
            except:
                pass
    
    atexit.register(cleanup)
    
    try:
        legal_ai()
    finally:
        cleanup()