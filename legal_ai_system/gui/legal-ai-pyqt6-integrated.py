# legal_ai_integrated.py - Complete integrated Legal AI Desktop Application

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from PyQt6 import QtWidgets, QtCore, QtGui

# Import all our custom modules
from legal_ai_desktop import (
    MainWindow, DocumentViewer, AnalyticsDashboard, 
    SettingsDialog, AboutDialog, LegalAIApplication,
    DocumentTableModel, Document
)
from legal_ai_widgets import (
    GlowingButton, FlipCard, TagCloud, TimelineWidget,
    NotificationWidget, SearchableComboBox, DockablePanel
)
from legal_ai_charts import (
    PieChartWidget, BarChartWidget, LineChartWidget, 
    HeatMapWidget, ChartData, AnalyticsDashboardWidget
)
from legal_ai_network import (
    NetworkManager, LegalAIAPIClient, DocumentProcessingWorker,
    WebSocketClient, AsyncAPIClient
)
from legal_ai_database import (
    DatabaseManager, CacheManager, PreferencesManager,
    DocumentSearchEngine
)


# ==================== INTEGRATED MAIN WINDOW ====================
class IntegratedMainWindow(QtWidgets.QMainWindow):
    """Enhanced main window with all integrated features"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Legal AI System - Professional Edition v3.0")
        self.setMinimumSize(1600, 1000)
        
        # Initialize components
        self.initializeServices()
        self.setupUI()
        self.setupDocks()
        self.setupMenuBar()
        self.setupToolBar()
        self.setupStatusBar()
        self.setupSystemTray()
        self.loadPreferences()
        self.connectSignals()
        
        # Apply theme
        self.applyTheme()
        
        # Start services
        QtCore.QTimer.singleShot(100, self.startServices)
        
    def initializeServices(self):
        """Initialize all backend services"""
        # Database
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager(self.db_manager)
        self.prefs_manager = PreferencesManager(self.db_manager)
        self.search_engine = DocumentSearchEngine(self.db_manager)
        
        # Network
        base_url = self.prefs_manager.get("api_base_url", "http://localhost:8000")
        self.network_manager = NetworkManager(base_url)
        self.api_client = LegalAIAPIClient(self.network_manager)
        self.processing_worker = DocumentProcessingWorker(self.api_client)
        
        # WebSocket for real-time updates
        ws_url = self.prefs_manager.get("websocket_url", "ws://localhost:8000/ws")
        self.websocket_client = WebSocketClient(ws_url)
        
        # Document storage
        self.documents: Dict[str, Document] = {}
        self.active_viewers: List[DocumentViewer] = []
        
    def setupUI(self):
        """Setup main UI"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # Top toolbar with quick actions
        top_toolbar = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_toolbar)
        top_layout.setContentsMargins(10, 5, 10, 5)
        
        # Search bar
        self.global_search = QtWidgets.QLineEdit()
        self.global_search.setPlaceholderText("Search documents, entities, or legal terms...")
        self.global_search.returnPressed.connect(self.performGlobalSearch)
        top_layout.addWidget(self.global_search)
        
        # Quick action buttons
        self.upload_btn = GlowingButton("Upload")
        self.upload_btn.clicked.connect(self.uploadDocuments)
        top_layout.addWidget(self.upload_btn)
        
        self.process_btn = GlowingButton("Process Queue")
        self.process_btn.clicked.connect(self.processQueue)
        top_layout.addWidget(self.process_btn)
        
        main_layout.addWidget(top_toolbar)
        
        # Main content area with tabs
        self.main_tabs = QtWidgets.QTabWidget()
        self.main_tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self.main_tabs.setMovable(True)
        
        # Dashboard tab
        self.dashboard_widget = self.createDashboard()
        self.main_tabs.addTab(self.dashboard_widget, "Dashboard")
        
        # Documents tab
        self.documents_widget = self.createDocumentsView()
        self.main_tabs.addTab(self.documents_widget, "Documents")
        
        # Analytics tab
        self.analytics_widget = AnalyticsDashboardWidget()
        self.main_tabs.addTab(self.analytics_widget, "Analytics")
        
        # Processing Queue tab
        self.queue_widget = self.createQueueView()
        self.main_tabs.addTab(self.queue_widget, "Processing Queue")
        
        main_layout.addWidget(self.main_tabs)
        
    def createDashboard(self) -> QtWidgets.QWidget:
        """Create dashboard with overview widgets"""
        dashboard = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(dashboard)
        
        # Stats cards
        stats_frame = QtWidgets.QFrame()
        stats_frame.setFrameStyle(QtWidgets.QFrame.Shape.Box)
        stats_layout = QtWidgets.QHBoxLayout(stats_frame)
        
        # Create flip cards for stats
        self.doc_count_card = FlipCard(
            "Total Documents\n0",
            "Click for details"
        )
        stats_layout.addWidget(self.doc_count_card)
        
        self.success_rate_card = FlipCard(
            "Success Rate\n0%",
            "Processing accuracy"
        )
        stats_layout.addWidget(self.success_rate_card)
        
        self.active_users_card = FlipCard(
            "Active Users\n0",
            "Currently online"
        )
        stats_layout.addWidget(self.active_users_card)
        
        layout.addWidget(stats_frame, 0, 0, 1, 2)
        
        # Recent activity timeline
        self.timeline = TimelineWidget()
        timeline_frame = QtWidgets.QGroupBox("Recent Activity")
        timeline_layout = QtWidgets.QVBoxLayout(timeline_frame)
        timeline_layout.addWidget(self.timeline)
        layout.addWidget(timeline_frame, 1, 0)
        
        # Tag cloud
        self.tag_cloud = TagCloud()
        tag_frame = QtWidgets.QGroupBox("Popular Tags")
        tag_layout = QtWidgets.QVBoxLayout(tag_frame)
        tag_layout.addWidget(self.tag_cloud)
        layout.addWidget(tag_frame, 1, 1)
        
        # Quick charts
        self.mini_pie = PieChartWidget()
        self.mini_pie.setMaximumHeight(300)
        layout.addWidget(self.mini_pie, 2, 0)
        
        self.mini_bar = BarChartWidget()
        self.mini_bar.setMaximumHeight(300)
        layout.addWidget(self.mini_bar, 2, 1)
        
        return dashboard
        
    def createDocumentsView(self) -> QtWidgets.QWidget:
        """Create documents management view"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Toolbar
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        
        # Filter controls
        toolbar_layout.addWidget(QtWidgets.QLabel("Status:"))
        self.status_filter = SearchableComboBox()
        self.status_filter.addItems(["All", "Pending", "Processing", "Completed", "Failed"])
        self.status_filter.currentTextChanged.connect(self.filterDocuments)
        toolbar_layout.addWidget(self.status_filter)
        
        toolbar_layout.addWidget(QtWidgets.QLabel("Type:"))
        self.type_filter = SearchableComboBox()
        self.type_filter.addItems(["All", "Contract", "Legal Brief", "Patent", "Compliance"])
        toolbar_layout.addWidget(self.type_filter)
        
        toolbar_layout.addStretch()
        
        # Export button
        export_btn = QtWidgets.QPushButton("Export Selected")
        export_btn.clicked.connect(self.exportDocuments)
        toolbar_layout.addWidget(export_btn)
        
        layout.addWidget(toolbar)
        
        # Document table
        self.doc_table = QtWidgets.QTableView()
        self.doc_model = DocumentTableModel()
        self.doc_table.setModel(self.doc_model)
        self.doc_table.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        self.doc_table.setAlternatingRowColors(True)
        self.doc_table.doubleClicked.connect(self.viewDocument)
        
        # Context menu
        self.doc_table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.doc_table.customContextMenuRequested.connect(self.showDocumentContextMenu)
        
        layout.addWidget(self.doc_table)
        
        return widget
        
    def createQueueView(self) -> QtWidgets.QWidget:
        """Create processing queue view"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Queue controls
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        
        pause_btn = QtWidgets.QPushButton("Pause Queue")
        pause_btn.setCheckable(True)
        pause_btn.toggled.connect(self.toggleQueueProcessing)
        controls_layout.addWidget(pause_btn)
        
        clear_btn = QtWidgets.QPushButton("Clear Completed")
        clear_btn.clicked.connect(self.clearCompletedItems)
        controls_layout.addWidget(clear_btn)
        
        controls_layout.addStretch()
        
        layout.addWidget(controls)
        
        # Queue list
        self.queue_list = QtWidgets.QListWidget()
        self.queue_list.setAlternatingRowColors(True)
        layout.addWidget(self.queue_list)
        
        return widget
        
    def setupDocks(self):
        """Setup dockable panels"""
        # File browser dock
        self.file_dock = DockablePanel("File Browser", self)
        file_tree = QtWidgets.QTreeView()
        file_model = QtWidgets.QFileSystemModel()
        file_model.setRootPath(str(Path.home()))
        file_tree.setModel(file_model)
        file_tree.setRootIndex(file_model.index(str(Path.home())))
        self.file_dock.setWidget(file_tree)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.file_dock)
        
        # Properties dock
        self.properties_dock = DockablePanel("Properties", self)
        props_widget = QtWidgets.QTextEdit()
        props_widget.setReadOnly(True)
        self.properties_dock.setWidget(props_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        
        # Console dock
        self.console_dock = DockablePanel("Console", self)
        self.console = QtWidgets.QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)
        self.console_dock.setWidget(self.console)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        
    def setupMenuBar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        upload_action = file_menu.addAction("Upload Documents...")
        upload_action.setShortcut("Ctrl+O")
        upload_action.triggered.connect(self.uploadDocuments)
        
        file_menu.addSeparator()
        
        import_action = file_menu.addAction("Import from Email...")
        export_action = file_menu.addAction("Export Results...")
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        settings_action = edit_menu.addAction("Preferences...")
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.showSettings)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        view_menu.addAction(self.file_dock.toggleViewAction())
        view_menu.addAction(self.properties_dock.toggleViewAction())
        view_menu.addAction(self.console_dock.toggleViewAction())
        
        view_menu.addSeparator()
        
        fullscreen_action = view_menu.addAction("Full Screen")
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggleFullScreen)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        batch_process_action = tools_menu.addAction("Batch Process...")
        ocr_tool_action = tools_menu.addAction("OCR Tool...")
        
        tools_menu.addSeparator()
        
        plugin_manager_action = tools_menu.addAction("Plugin Manager...")
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        docs_action = help_menu.addAction("Documentation")
        docs_action.setShortcut("F1")
        
        help_menu.addSeparator()
        
        about_action = help_menu.addAction("About...")
        about_action.triggered.connect(self.showAbout)
        
    def setupToolBar(self):
        """Setup main toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        
        # Create actions with icons
        upload_action = toolbar.addAction("Upload")
        process_action = toolbar.addAction("Process")
        
        toolbar.addSeparator()
        
        view_action = toolbar.addAction("View")
        analytics_action = toolbar.addAction("Analytics")
        
        toolbar.addSeparator()
        
        settings_action = toolbar.addAction("Settings")
        
    def setupStatusBar(self):
        """Setup status bar with widgets"""
        self.status_bar = self.statusBar()
        
        # Status message
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addWidget(self.progress_bar)
        
        # Permanent widgets
        self.doc_count_label = QtWidgets.QLabel("Documents: 0")
        self.status_bar.addPermanentWidget(self.doc_count_label)
        
        self.connection_indicator = QtWidgets.QLabel("● Offline")
        self.connection_indicator.setStyleSheet("color: #f44336;")
        self.status_bar.addPermanentWidget(self.connection_indicator)
        
        self.user_label = QtWidgets.QLabel("User: Admin")
        self.status_bar.addPermanentWidget(self.user_label)
        
    def setupSystemTray(self):
        """Setup system tray integration"""
        self.tray_icon = QtWidgets.QSystemTrayIcon(self)
        self.tray_icon.setToolTip("Legal AI System")
        
        # Create tray menu
        tray_menu = QtWidgets.QMenu()
        
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)
        
        hide_action = tray_menu.addAction("Hide")
        hide_action.triggered.connect(self.hide)
        
        tray_menu.addSeparator()
        
        process_action = tray_menu.addAction("Process Queue")
        process_action.triggered.connect(self.processQueue)
        
        tray_menu.addSeparator()
        
        exit_action = tray_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.trayIconActivated)
        self.tray_icon.show()
        
    def connectSignals(self):
        """Connect all signals"""
        # Database signals
        self.db_manager.databaseReady.connect(self.onDatabaseReady)
        self.db_manager.error.connect(self.onDatabaseError)
        
        # Network signals
        self.network_manager.connectionStatusChanged.connect(self.onConnectionStatusChanged)
        self.api_client.documentsLoaded.connect(self.onDocumentsLoaded)
        self.api_client.documentUploaded.connect(self.onDocumentUploaded)
        self.api_client.processingComplete.connect(self.onProcessingComplete)
        
        # WebSocket signals
        self.websocket_client.connected.connect(self.onWebSocketConnected)
        self.websocket_client.messageReceived.connect(self.onWebSocketMessage)
        
        # Worker signals
        self.processing_worker.progress.connect(self.onProcessingProgress)
        
        # Preferences signals
        self.prefs_manager.preferenceChanged.connect(self.onPreferenceChanged)
        
    def applyTheme(self):
        """Apply application theme"""
        theme = self.prefs_manager.get("theme", "dark")
        
        if theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1a1a1a;
                }
                QWidget {
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                QGroupBox {
                    border: 2px solid #444;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QTabWidget::pane {
                    border: 1px solid #444;
                    background-color: #2d2d2d;
                }
                QTabBar::tab {
                    background-color: #3d3d3d;
                    color: white;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #dc143c;
                }
                QTableView {
                    background-color: #2d2d2d;
                    alternate-background-color: #3d3d3d;
                    gridline-color: #444;
                }
                QHeaderView::section {
                    background-color: #3d3d3d;
                    color: white;
                    padding: 5px;
                    border: none;
                }
            """)
            
    def startServices(self):
        """Start background services"""
        self.log("Starting services...")
        
        # Connect to backend
        self.network_manager.checkConnection()
        
        # Connect WebSocket
        self.websocket_client.connect()
        
        # Load initial data
        self.loadDocuments()
        self.updateDashboard()
        
    # ==================== SLOTS ====================
    def onDatabaseReady(self):
        """Handle database ready"""
        self.log("Database initialized")
        self.loadLocalDocuments()
        
    def onDatabaseError(self, error: str):
        """Handle database error"""
        self.log(f"Database error: {error}", level="error")
        QtWidgets.QMessageBox.critical(self, "Database Error", error)
        
    def onConnectionStatusChanged(self, connected: bool):
        """Handle connection status change"""
        if connected:
            self.connection_indicator.setText("● Online")
            self.connection_indicator.setStyleSheet("color: #4caf50;")
            self.showNotification("Connected to server", "success")
        else:
            self.connection_indicator.setText("● Offline")
            self.connection_indicator.setStyleSheet("color: #f44336;")
            self.showNotification("Disconnected from server", "error")
            
    def onDocumentsLoaded(self, documents: List[Dict]):
        """Handle documents loaded from API"""
        self.log(f"Loaded {len(documents)} documents from server")
        
        # Update model
        for doc_data in documents:
            doc = Document(
                id=doc_data["document_id"],
                filename=doc_data["filename"],
                status=doc_data["status"],
                progress=doc_data.get("progress", 0),
                uploaded_at=datetime.fromisoformat(doc_data["uploaded_at"]),
                file_size=doc_data.get("file_size", 0),
                doc_type=doc_data.get("type", "Unknown")
            )
            self.documents[doc.id] = doc
            self.doc_model.addDocument(doc)
            
            # Save to local database
            self.db_manager.saveDocument(
                doc.id, doc.filename,
                file_size=doc.file_size,
                metadata=doc_data.get("metadata")
            )
            
    def onDocumentUploaded(self, doc_id: str):
        """Handle document uploaded"""
        self.log(f"Document uploaded: {doc_id}")
        self.showNotification("Document uploaded successfully", "success")
        
        # Add to processing queue
        if doc_id in self.documents:
            self.queue_list.addItem(f"Processing: {self.documents[doc_id].filename}")
            
    def onProcessingComplete(self, doc_id: str, results: Dict):
        """Handle processing complete"""
        self.log(f"Processing complete for document: {doc_id}")
        
        # Update document status
        if doc_id in self.documents:
            self.documents[doc_id].status = "completed"
            self.doc_model.updateDocument(doc_id, "completed", 1.0)
            
        # Save results to database
        self.db_manager.updateDocumentStatus(doc_id, "completed", results)
        
        # Index for search
        if "text_content" in results:
            self.search_engine.indexDocument(
                doc_id,
                self.documents[doc_id].filename,
                results["text_content"],
                results
            )
            
        self.showNotification(f"Processing complete: {self.documents[doc_id].filename}", "success")
        
    def onProcessingProgress(self, doc_id: str, progress: int, stage: str):
        """Handle processing progress"""
        self.log(f"Processing {doc_id}: {stage} ({progress}%)")
        
        if doc_id in self.documents:
            self.doc_model.updateDocument(doc_id, "processing", progress / 100)
            
    def onWebSocketConnected(self):
        """Handle WebSocket connected"""
        self.log("WebSocket connected")
        
    def onWebSocketMessage(self, message: Dict):
        """Handle WebSocket message"""
        msg_type = message.get("type")
        
        if msg_type == "document_update":
            doc_id = message.get("document_id")
            status = message.get("status")
            self.doc_model.updateDocument(doc_id, status, message.get("progress", 0))
            
        elif msg_type == "notification":
            self.showNotification(message.get("text", ""), message.get("level", "info"))
            
    def onPreferenceChanged(self, key: str, value: Any):
        """Handle preference changed"""
        if key == "theme":
            self.applyTheme()
            
    # ==================== ACTIONS ====================
    def uploadDocuments(self):
        """Upload documents"""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Documents",
            str(Path.home()),
            "Documents (*.pdf *.docx *.txt *.md);;All Files (*.*)"
        )
        
        if files:
            self.upload_btn.startGlow()
            
            for file_path in files:
                path = Path(file_path)
                
                # Upload via API
                self.api_client.uploadDocument(
                    path,
                    {
                        "enable_ner": self.prefs_manager.get("enable_ner", True),
                        "enable_llm": self.prefs_manager.get("enable_llm", True),
                        "confidence_threshold": self.prefs_manager.get("confidence_threshold", 0.7)
                    }
                )
                
            self.upload_btn.stopGlow()
            
    def processQueue(self):
        """Process document queue"""
        self.process_btn.startGlow()
        
        # Start processing worker
        for doc_id, doc in self.documents.items():
            if doc.status == "pending":
                self.processing_worker.addDocument(
                    doc_id,
                    Path(doc.filename),
                    {}
                )
                
    def viewDocument(self, index: QtCore.QModelIndex):
        """View document details"""
        row = index.row()
        doc_id = self.doc_model.documents.iloc[row]["ID"]
        
        viewer = DocumentViewer(doc_id, self)
        viewer.show()
        self.active_viewers.append(viewer)
        
    def performGlobalSearch(self):
        """Perform global search"""
        query = self.global_search.text()
        if query:
            results = self.search_engine.search(query)
            self.log(f"Search returned {len(results)} results")
            
            # Show results in new tab
            results_widget = QtWidgets.QListWidget()
            for result in results:
                item = QtWidgets.QListWidgetItem(f"{result['filename']}: {result['snippet']}")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, result['document_id'])
                results_widget.addItem(item)
                
            self.main_tabs.addTab(results_widget, f"Search: {query}")
            self.main_tabs.setCurrentWidget(results_widget)
            
    def showSettings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Save settings
            self.prefs_manager.set("api_url", dialog.api_url.text())
            self.prefs_manager.set("openai_key", dialog.openai_key.text())
            self.prefs_manager.set("enable_ner", dialog.enable_ner.isChecked())
            self.prefs_manager.set("enable_llm", dialog.enable_llm.isChecked())
            
    def showAbout(self):
        """Show about dialog"""
        dialog = AboutDialog(self)
        dialog.exec()
        
    def showNotification(self, message: str, notification_type: str = "info"):
        """Show notification popup"""
        notif = NotificationWidget(message, notification_type)
        notif.show(self)
        
    def log(self, message: str, level: str = "info"):
        """Log message to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == "error":
            html = f'<span style="color: #f44336;">[{timestamp}] ERROR: {message}</span>'
        elif level == "warning":
            html = f'<span style="color: #ff9800;">[{timestamp}] WARN: {message}</span>'
        else:
            html = f'<span style="color: #4caf50;">[{timestamp}] INFO: {message}</span>'
            
        self.console.append(html)
        
    def updateDashboard(self):
        """Update dashboard widgets"""
        # Update flip cards
        doc_count = len(self.documents)
        self.doc_count_card.front_content = f"Total Documents\n{doc_count}"
        
        # Update timeline
        now = QtCore.QDateTime.currentDateTime()
        for i, (doc_id, doc) in enumerate(list(self.documents.items())[:5]):
            self.timeline.addEvent(
                now.addSecs(-i * 3600),
                doc.filename,
                f"Status: {doc.status}",
                "success" if doc.status == "completed" else "info"
            )
            
        # Update tag cloud
        tags = [
            {"text": "Contract", "weight": 2.0},
            {"text": "Legal", "weight": 1.5},
            {"text": "Compliance", "weight": 1.0},
            {"text": "Patent", "weight": 0.8},
        ]
        self.tag_cloud.setTags(tags)
        
        # Update mini charts
        pie_data = [
            ChartData("Completed", 75),
            ChartData("Processing", 15),
            ChartData("Pending", 10),
        ]
        self.mini_pie.setData(pie_data)
        
        bar_data = [
            ChartData("Mon", 12),
            ChartData("Tue", 19),
            ChartData("Wed", 15),
            ChartData("Thu", 25),
            ChartData("Fri", 22),
        ]
        self.mini_bar.setData(bar_data)
        
    def loadLocalDocuments(self):
        """Load documents from local database"""
        docs = self.db_manager.getDocuments(limit=1000)
        for doc_data in docs:
            doc = Document(
                id=doc_data["document_id"],
                filename=doc_data["filename"],
                status=doc_data["status"],
                progress=1.0 if doc_data["status"] == "completed" else 0.5,
                uploaded_at=datetime.fromisoformat(doc_data["upload_date"]),
                file_size=doc_data.get("file_size", 0)
            )
            self.documents[doc.id] = doc
            self.doc_model.addDocument(doc)
            
    def loadDocuments(self):
        """Load documents from API"""
        self.api_client.loadDocuments()
        
    def filterDocuments(self):
        """Filter documents based on criteria"""
        # Implement filtering logic
        pass
        
    def exportDocuments(self):
        """Export selected documents"""
        # Implement export logic
        pass
        
    def showDocumentContextMenu(self, pos: QtCore.QPoint):
        """Show context menu for documents"""
        menu = QtWidgets.QMenu(self)
        
        view_action = menu.addAction("View")
        view_action.triggered.connect(lambda: self.viewDocument(self.doc_table.currentIndex()))
        
        menu.addSeparator()
        
        reprocess_action = menu.addAction("Reprocess")
        delete_action = menu.addAction("Delete")
        
        menu.exec(self.doc_table.mapToGlobal(pos))
        
    def toggleQueueProcessing(self, paused: bool):
        """Toggle queue processing"""
        if paused:
            self.processing_worker.requestInterruption()
        else:
            self.processQueue()
            
    def clearCompletedItems(self):
        """Clear completed items from queue"""
        for i in range(self.queue_list.count() - 1, -1, -1):
            item = self.queue_list.item(i)
            if "Complete" in item.text():
                self.queue_list.takeItem(i)
                
    def toggleFullScreen(self):
        """Toggle full screen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
            
    def trayIconActivated(self, reason):
        """Handle tray icon activation"""
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.raise_()
                self.activateWindow()
                
    def loadPreferences(self):
        """Load user preferences"""
        # Window geometry
        geometry = self.prefs_manager.get("window_geometry")
        if geometry:
            self.restoreGeometry(QtCore.QByteArray.fromBase64(geometry.encode()))
            
        # Window state
        state = self.prefs_manager.get("window_state")
        if state:
            self.restoreState(QtCore.QByteArray.fromBase64(state.encode()))
            
    def savePreferences(self):
        """Save user preferences"""
        # Window geometry
        self.prefs_manager.set(
            "window_geometry",
            bytes(self.saveGeometry().toBase64()).decode()
        )
        
        # Window state
        self.prefs_manager.set(
            "window_state",
            bytes(self.saveState().toBase64()).decode()
        )
        
    def closeEvent(self, event):
        """Handle close event"""
        # Save preferences
        self.savePreferences()
        
        # Close child windows
        for viewer in self.active_viewers:
            viewer.close()
            
        # Cleanup
        self.websocket_client.disconnect()
        self.processing_worker.quit()
        self.processing_worker.wait()
        
        # Ask confirmation
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


# ==================== MAIN ====================
def main():
    # Create application
    app = LegalAIApplication(sys.argv)
    
    # Create splash screen
    splash = QtWidgets.QSplashScreen()
    splash.setPixmap(QtGui.QPixmap(600, 400))
    splash.showMessage(
        "Loading Legal AI System...",
        QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignCenter,
        QtCore.Qt.GlobalColor.white
    )
    splash.show()
    
    # Process events
    app.processEvents()
    
    # Create main window
    window = IntegratedMainWindow()
    
    # Show window
    QtCore.QTimer.singleShot(2000, splash.close)
    QtCore.QTimer.singleShot(2000, window.show)
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
