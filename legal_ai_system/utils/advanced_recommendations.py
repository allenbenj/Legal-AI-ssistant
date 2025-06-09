"""
Advanced Recommendations for Document Processing System

Additional architectural improvements and best practices.
"""

# === 1. INTELLIGENT ROUTING & LOAD BALANCING ===

class DocumentRouter:
    """Intelligent routing based on document characteristics"""
    
    def __init__(self, processor_pool: Dict[str, List[BaseProcessor]]):
        self.processor_pool = processor_pool
        self.routing_rules = self._initialize_routing_rules()
        self.load_tracker = LoadTracker()
    
    def _initialize_routing_rules(self) -> Dict[str, Any]:
        """Define routing rules based on document characteristics"""
        return {
            "size_based": {
                "small": {"max_size_mb": 1, "preferred_processors": ["fast"]},
                "medium": {"max_size_mb": 10, "preferred_processors": ["standard"]},
                "large": {"max_size_mb": 100, "preferred_processors": ["heavy"]}
            },
            "complexity_based": {
                "simple": {"indicators": ["memo", "letter"], "preferred_model": "gpt-3.5-turbo"},
                "moderate": {"indicators": ["contract", "agreement"], "preferred_model": "gpt-4"},
                "complex": {"indicators": ["brief", "statute"], "preferred_model": "gpt-4-turbo"}
            },
            "priority_based": {
                "urgent": {"sla_minutes": 5, "dedicated_workers": 2},
                "normal": {"sla_minutes": 30, "dedicated_workers": 1},
                "batch": {"sla_minutes": 240, "dedicated_workers": 0}
            }
        }
    
    async def route_document(self, document: Path, metadata: Dict[str, Any]) -> BaseProcessor:
        """Route document to optimal processor"""
        # Analyze document characteristics
        size_category = self._categorize_by_size(document)
        complexity = self._assess_complexity(document, metadata)
        priority = metadata.get("priority", "normal")
        
        # Select optimal processor
        processor_type = self._select_processor_type(size_category, complexity, priority)
        processor = self._get_least_loaded_processor(processor_type)
        
        # Track routing decision
        self.load_tracker.record_routing(processor, document, {
            "size_category": size_category,
            "complexity": complexity,
            "priority": priority
        })
        
        return processor
    
    def _get_least_loaded_processor(self, processor_type: str) -> BaseProcessor:
        """Get processor with lowest current load"""
        processors = self.processor_pool.get(processor_type, [])
        if not processors:
            raise ValueError(f"No processors available for type: {processor_type}")
        
        return min(processors, key=lambda p: self.load_tracker.get_load(p))

# === 2. ADVANCED CACHING WITH EMBEDDING SIMILARITY ===

class SemanticCache:
    """Cache with semantic similarity for near-duplicate detection"""
    
    def __init__(self, embedding_model: Any, similarity_threshold: float = 0.95):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.embeddings = {}  # document_id -> embedding
        self.cache = {}  # document_id -> result
        
    async def get_similar(self, content: str) -> Optional[ProcessingResult]:
        """Find similar cached document using embeddings"""
        query_embedding = await self._get_embedding(content)
        
        best_match = None
        best_similarity = 0.0
        
        for doc_id, cached_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = doc_id
        
        if best_match:
            return self.cache.get(best_match)
        
        return None
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text content"""
        # Truncate to model's max length
        truncated = text[:8000]
        return await self.embedding_model.encode(truncated)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === 3. REAL-TIME PROCESSING FEEDBACK ===

class ProcessingMonitor:
    """Real-time monitoring with WebSocket support"""
    
    def __init__(self):
        self.active_jobs = {}
        self.subscribers = set()
        self.metrics_buffer = deque(maxlen=1000)
    
    async def track_job(self, job_id: str, document: str):
        """Start tracking a processing job"""
        self.active_jobs[job_id] = {
            "document": document,
            "status": "started",
            "progress": 0,
            "stages": {},
            "started_at": datetime.now(),
            "events": []
        }
        
        await self._broadcast_update(job_id, "job_started")
    
    async def update_progress(self, job_id: str, stage: str, progress: float, 
                            details: Dict[str, Any] = None):
        """Update job progress"""
        if job_id not in self.active_jobs:
            return
        
        job = self.active_jobs[job_id]
        job["stages"][stage] = {
            "progress": progress,
            "details": details or {},
            "updated_at": datetime.now()
        }
        
        # Calculate overall progress
        total_stages = 4  # file_processing, entity_extraction, validation, storage
        completed_stages = sum(1 for s in job["stages"].values() if s["progress"] >= 1.0)
        job["progress"] = (completed_stages / total_stages) * 100
        
        await self._broadcast_update(job_id, "progress_update", {
            "stage": stage,
            "progress": progress,
            "overall_progress": job["progress"]
        })
    
    async def _broadcast_update(self, job_id: str, event_type: str, data: Dict = None):
        """Broadcast update to all subscribers"""
        message = {
            "job_id": job_id,
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        # Store in metrics buffer
        self.metrics_buffer.append(message)
        
        # Broadcast to WebSocket subscribers
        for subscriber in self.subscribers:
            try:
                await subscriber.send(json.dumps(message))
            except Exception as e:
                logging.error(f"Failed to send update to subscriber: {e}")

# === 4. BATCH PROCESSING OPTIMIZATION ===

class BatchProcessor:
    """Optimized batch processing with intelligent grouping"""
    
    def __init__(self, pipeline: DocumentProcessingPipeline):
        self.pipeline = pipeline
        self.batch_queue = asyncio.Queue()
        self.processing = False
    
    async def add_batch(self, documents: List[Path], options: Dict[str, Any] = None):
        """Add documents for batch processing"""
        # Group documents by characteristics
        groups = self._group_documents(documents)
        
        for group_key, group_docs in groups.items():
            batch = {
                "id": f"batch_{datetime.now().timestamp()}",
                "documents": group_docs,
                "options": options or {},
                "group_key": group_key,
                "created_at": datetime.now()
            }
            await self.batch_queue.put(batch)
    
    def _group_documents(self, documents: List[Path]) -> Dict[str, List[Path]]:
        """Group documents by processing characteristics"""
        groups = {}
        
        for doc in documents:
            # Group by file type and size
            file_type = doc.suffix.lower()
            size_mb = doc.stat().st_size / (1024 * 1024)
            size_category = "small" if size_mb < 1 else "medium" if size_mb < 10 else "large"
            
            group_key = f"{file_type}_{size_category}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(doc)
        
        return groups
    
    async def process_batches(self):
        """Process batches with optimized resource usage"""
        self.processing = True
        
        while self.processing:
            try:
                # Get next batch
                batch = await asyncio.wait_for(self.batch_queue.get(), timeout=5.0)
                
                # Process based on group characteristics
                if "large" in batch["group_key"]:
                    # Process large files sequentially
                    results = await self._process_sequential(batch)
                else:
                    # Process small/medium files in parallel
                    results = await self._process_parallel(batch)
                
                # Store results
                await self._store_batch_results(batch["id"], results)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Batch processing error: {e}")

# === 5. ML-POWERED OPTIMIZATION ===

class MLOptimizer:
    """Machine learning optimizer for processing parameters"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=10000)
        self.parameter_optimizer = self._initialize_optimizer()
    
    def record_performance(self, document_type: str, parameters: Dict[str, Any], 
                          metrics: Dict[str, float]):
        """Record processing performance for ML training"""
        self.performance_history.append({
            "document_type": document_type,
            "parameters": parameters,
            "metrics": metrics,
            "timestamp": datetime.now()
        })
    
    async def get_optimal_parameters(self, document_type: str, 
                                   document_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML-optimized processing parameters"""
        # Default parameters
        params = {
            "chunk_size": 3000,
            "chunk_overlap": 200,
            "temperature": 0.1,
            "max_tokens": 2000,
            "confidence_threshold": 0.7
        }
        
        # Apply ML optimizations if enough history
        if len(self.performance_history) > 100:
            similar_docs = self._find_similar_documents(document_type, document_features)
            
            if similar_docs:
                # Analyze best performing parameters
                best_params = self._analyze_best_parameters(similar_docs)
                params.update(best_params)
        
        return params
    
    def _analyze_best_parameters(self, similar_docs: List[Dict]) -> Dict[str, Any]:
        """Analyze parameters from best performing similar documents"""
        # Sort by performance metric (e.g., processing time, accuracy)
        sorted_docs = sorted(similar_docs, 
                           key=lambda x: x["metrics"].get("f1_score", 0) / 
                                       x["metrics"].get("processing_time", 1),
                           reverse=True)
        
        # Take top 10% performers
        top_performers = sorted_docs[:max(1, len(sorted_docs) // 10)]
        
        # Average their parameters
        optimal_params = {}
        param_names = ["chunk_size", "temperature", "confidence_threshold"]
        
        for param in param_names:
            values = [d["parameters"].get(param) for d in top_performers if param in d["parameters"]]
            if values:
                optimal_params[param] = sum(values) / len(values)
        
        return optimal_params

# === 6. SECURITY & COMPLIANCE ===

class SecurityManager:
    """Security and compliance management"""
    
    def __init__(self):
        self.encryption_key = self._load_encryption_key()
        self.audit_logger = AuditLogger()
        self.pii_detector = PIIDetector()
    
    async def secure_document(self, document: ProcessingResult) -> ProcessingResult:
        """Apply security measures to document"""
        # Detect and handle PII
        if document.content and "text" in document.content:
            pii_results = await self.pii_detector.scan(document.content["text"])
            
            if pii_results.has_pii:
                # Redact or encrypt PII based on policy
                document = await self._handle_pii(document, pii_results)
        
        # Encrypt sensitive fields
        if self._is_sensitive(document):
            document = await self._encrypt_sensitive_fields(document)
        
        # Audit log
        await self.audit_logger.log_access(document, "processed")
        
        return document
    
    async def _handle_pii(self, document: ProcessingResult, 
                         pii_results: PIIResults) -> ProcessingResult:
        """Handle PII based on compliance requirements"""
        policy = self._get_pii_policy(document.metadata.get("jurisdiction", "US"))
        
        if policy == "redact":
            # Redact PII from text
            document.content["text"] = pii_results.redacted_text
            document.metadata["pii_redacted"] = True
        elif policy == "encrypt":
            # Encrypt PII data
            for pii_item in pii_results.pii_items:
                encrypted = self._encrypt(pii_item.value)
                document.metadata[f"encrypted_{pii_item.type}"] = encrypted
        
        return document

# === 7. ADVANCED ERROR RECOVERY ===

class SmartErrorRecovery:
    """Intelligent error recovery with multiple strategies"""
    
    def __init__(self):
        self.recovery_strategies = {
            "file_corrupted": self._recover_corrupted_file,
            "llm_timeout": self._recover_llm_timeout,
            "memory_error": self._recover_memory_error,
            "parsing_error": self._recover_parsing_error
        }
        self.recovery_history = deque(maxlen=1000)
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from error"""
        error_type = self._classify_error(error)
        
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            try:
                result = await strategy(error, context)
                self._record_recovery(error_type, True, context)
                return result
            except Exception as recovery_error:
                self._record_recovery(error_type, False, context)
                raise
        else:
            # Unknown error type - try generic recovery
            return await self._generic_recovery(error, context)
    
    async def _recover_corrupted_file(self, error: Exception, context: Dict[str, Any]):
        """Recover from corrupted file errors"""
        file_path = context.get("file_path")
        
        # Try alternative readers
        alternatives = [
            ("textract", self._try_textract),
            ("pandoc", self._try_pandoc),
            ("raw_text", self._try_raw_text)
        ]
        
        for name, method in alternatives:
            try:
                return await method(file_path)
            except:
                continue
        
        raise error
    
    async def _recover_llm_timeout(self, error: Exception, context: Dict[str, Any]):
        """Recover from LLM timeout"""
        # Try with smaller chunks
        original_chunk_size = context.get("chunk_size", 3000)
        new_chunk_size = original_chunk_size // 2
        
        if new_chunk_size < 500:
            raise error
        
        context["chunk_size"] = new_chunk_size
        context["retry_with_smaller_chunks"] = True
        
        # Retry with modified context
        return await context["retry_func"](context)

# === 8. PERFORMANCE PROFILING ===

class PerformanceProfiler:
    """Detailed performance profiling for optimization"""
    
    def __init__(self):
        self.profiles = {}
        self.bottlenecks = deque(maxlen=100)
    
    @contextmanager
    def profile_section(self, section_name: str):
        """Profile a code section"""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            profile = {
                "section": section_name,
                "duration": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "timestamp": datetime.now()
            }
            
            # Store profile
            if section_name not in self.profiles:
                self.profiles[section_name] = deque(maxlen=1000)
            self.profiles[section_name].append(profile)
            
            # Check for bottlenecks
            if profile["duration"] > 5.0:  # More than 5 seconds
                self.bottlenecks.append(profile)
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        if not self.profiles:
            return {"status": "no_data"}
        
        analysis = {
            "slowest_sections": [],
            "memory_intensive_sections": [],
            "optimization_suggestions": []
        }
        
        # Find slowest sections
        for section, profiles in self.profiles.items():
            if profiles:
                avg_duration = sum(p["duration"] for p in profiles) / len(profiles)
                if avg_duration > 1.0:
                    analysis["slowest_sections"].append({
                        "section": section,
                        "avg_duration": avg_duration,
                        "samples": len(profiles)
                    })
        
        # Generate suggestions
        for slow_section in analysis["slowest_sections"]:
            if "entity_extraction" in slow_section["section"]:
                analysis["optimization_suggestions"].append(
                    "Consider using a smaller model for initial entity detection"
                )
            elif "file_processing" in slow_section["section"]:
                analysis["optimization_suggestions"].append(
                    "Enable caching for frequently processed file types"
                )
        
        return analysis

# === 9. API VERSIONING & BACKWARD COMPATIBILITY ===

class APIVersionManager:
    """Manage API versions for backward compatibility"""
    
    def __init__(self):
        self.versions = {
            "v1": V1ProcessingAPI(),
            "v2": V2ProcessingAPI(),
            "v3": V3ProcessingAPI()  # Latest
        }
        self.default_version = "v3"
    
    async def process_request(self, request: Dict[str, Any], version: str = None) -> Dict[str, Any]:
        """Process request with appropriate API version"""
        version = version or self.default_version
        
        if version not in self.versions:
            raise ValueError(f"Unsupported API version: {version}")
        
        api = self.versions[version]
        
        # Transform request to latest format if needed
        if version != self.default_version:
            request = self._transform_request(request, version, self.default_version)
        
        # Process with latest implementation
        result = await self.versions[self.default_version].process(request)
        
        # Transform response back to requested version format
        if version != self.default_version:
            result = self._transform_response(result, self.default_version, version)
        
        return result

# === 10. DEVELOPER EXPERIENCE ===


# === INTEGRATION EXAMPLE ===

async def advanced_pipeline_example():
    """Example of using advanced features together"""
    
    # Initialize components
    config = ProcessingConfig(Path("config.yaml"))
    llm_manager = LLMProviderManager()
    
    # Core pipeline
    pipeline = DocumentProcessingPipeline(config.get("services"), llm_manager)
    
    # Advanced components
    router = DocumentRouter({"standard": [pipeline]})
    monitor = ProcessingMonitor()
    profiler = PerformanceProfiler()
    security = SecurityManager()
    ml_optimizer = MLOptimizer()
    batch_processor = BatchProcessor(pipeline)
    
    # Process with advanced features
    documents = Path("./legal_documents").glob("*.pdf")
    
    for doc in documents:
        # Get ML-optimized parameters
        doc_features = {"size_mb": doc.stat().st_size / (1024**2)}
        optimal_params = await ml_optimizer.get_optimal_parameters("pdf", doc_features)
        
        # Route to optimal processor
        processor = await router.route_document(doc, {"priority": "normal"})
        
        # Process with monitoring
        job_id = f"job_{doc.stem}"
        await monitor.track_job(job_id, str(doc))
        
        with profiler.profile_section(f"process_{doc.name}"):
            # Process document
            result = await processor.process(doc, optimal_params)
            
            # Apply security
            result = await security.secure_document(result)
        
        # Update monitoring
        await monitor.update_progress(job_id, "completed", 1.0)
        
        # Record performance for ML
        ml_optimizer.record_performance(
            "pdf", 
            optimal_params,
            {"processing_time": result.processing_time, "entities_found": len(result.entities or [])}
        )
    
    # Get performance analysis
    bottlenecks = profiler.get_bottleneck_analysis()
    print(f"Performance bottlenecks: {bottlenecks}")
