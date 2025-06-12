# legal_ai_integrated.py - Complete integrated Legal AI Desktop Application

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

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

from .sections.dashboard_section import DashboardSection
from .sections.document_section import DocumentSection
from .sections.queue_section import QueueSection


# ==================== INTEGRATED MAIN WINDOW ====================
class IntegratedMainWindow(QMainWindow):
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
        QTimer.singleShot(100, self.startServices)
        
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
        

        
    def setupUI(self):
        """Setup main UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Instantiate helper sections
        self.dashboard_section = DashboardSection(self)
        self.document_section = DocumentSection(self)
        self.queue_section = QueueSection(self)
        
        # Top toolbar with quick actions
        top_toolbar = QWidget()
        top_layout = QHBoxLayout(top_toolbar)
        top_layout.setContentsMargins(10, 5, 10, 5)
        
        # Search bar
        self.global_search = QLineEdit()
        self.global_search.setPlaceholderText(
            "Search documents, entities, or legal terms..."
        )
        # Delegate search handling to document section
        self.global_search.returnPressed.connect(
            lambda: self.document_section.perform_global_search()
        )
        top_layout.addWidget(self.global_search)
        
        # Quick action buttons
        self.upload_btn = GlowingButton("Upload")
        self.upload_btn.clicked.connect(self.document_section.upload_documents)
        top_layout.addWidget(self.upload_btn)
        
        self.process_btn = GlowingButton("Process Queue")
        self.process_btn.clicked.connect(self.queue_section.process_queue)
        top_layout.addWidget(self.process_btn)
        
        main_layout.addWidget(top_toolbar)
        
        # Main content area with tabs
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.main_tabs.setMovable(True)
        
        # Dashboard tab
        self.dashboard_widget = self.dashboard_section.create_widget()
        self.main_tabs.addTab(self.dashboard_widget, "Dashboard")
        
        # Documents tab
        self.documents_widget = self.document_section.create_widget()
        self.main_tabs.addTab(self.documents_widget, "Documents")
        
        # Analytics tab
        self.analytics_widget = AnalyticsDashboardWidget()
        self.main_tabs.addTab(self.analytics_widget, "Analytics")
        
        # Processing Queue tab
        self.queue_widget = self.queue_section.create_widget()
        self.main_tabs.addTab(self.queue_widget, "Processing Queue")
        
        main_layout.addWidget(self.main_tabs)
        

        
    def setupDocks(self):
        """Setup dockable panels"""
        # File browser dock
        self.file_dock = DockablePanel("File Browser", self)
        file_tree = QTreeView()
        file_model = QFileSystemModel()
        file_model.setRootPath(str(Path.home()))
        file_tree.setModel(file_model)
        file_tree.setRootIndex(file_model.index(str(Path.home())))
        self.file_dock.setWidget(file_tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.file_dock)
        
        # Properties dock
        self.properties_dock = DockablePanel("Properties", self)
        props_widget = QTextEdit()
        props_widget.setReadOnly(True)
        self.properties_dock.setWidget(props_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        
        # Console dock
        self.console_dock = DockablePanel("Console", self)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)
        self.console_dock.setWidget(self.console)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        
    def setupMenuBar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        upload_action = file_menu.addAction("Upload Documents...")
        upload_action.setShortcut("Ctrl+O")
        upload_action.triggered.connect(self.document_section.upload_documents)
        
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
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        
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
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addWidget(self.progress_bar)
        
        # Permanent widgets
        self.doc_count_label = QLabel("Documents: 0")
        self.status_bar.addPermanentWidget(self.doc_count_label)
        
        self.connection_indicator = QLabel("● Offline")
        self.connection_indicator.setStyleSheet("color: #f44336;")
        self.status_bar.addPermanentWidget(self.connection_indicator)
        
        self.user_label = QLabel("User: Admin")
        self.status_bar.addPermanentWidget(self.user_label)
        
    def setupSystemTray(self):
        """Setup system tray integration"""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setToolTip("Legal AI System")
        
        # Create tray menu
        tray_menu = QMenu()
        
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)
        
        hide_action = tray_menu.addAction("Hide")
        hide_action.triggered.connect(self.hide)
        
        tray_menu.addSeparator()
        
        process_action = tray_menu.addAction("Process Queue")
        process_action.triggered.connect(self.queue_section.process_queue)
        
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
        self.network_manager.connectionStatusChanged.connect(
            self.onConnectionStatusChanged
        )
        self.api_client.documentsLoaded.connect(
            self.document_section.on_documents_loaded
        )
        self.api_client.documentUploaded.connect(
            self.document_section.on_document_uploaded
        )
        self.api_client.processingComplete.connect(
            self.document_section.on_processing_complete
        )
        
        # WebSocket signals
        self.websocket_client.connected.connect(self.onWebSocketConnected)
        self.websocket_client.messageReceived.connect(self.onWebSocketMessage)
        
        # Worker signals
        self.processing_worker.progress.connect(
            self.document_section.on_processing_progress
        )
        
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
        self.document_section.load_documents()
        self.dashboard_section.update()
        
    # ==================== SLOTS ====================
    def onDatabaseReady(self):
        """Handle database ready"""
        self.log("Database initialized")
        self.document_section.load_local_documents()
        
    def onDatabaseError(self, error: str):
        """Handle database error"""
        self.log(f"Database error: {error}", level="error")
        QMessageBox.critical(self, "Database Error", error)
        
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
            
            
    def onWebSocketConnected(self):
        """Handle WebSocket connected"""
        self.log("WebSocket connected")
        
    def onWebSocketMessage(self, message: Dict):
        """Handle WebSocket message"""
        msg_type = message.get("type")
        
        if msg_type == "document_update":
            doc_id = message.get("document_id")
            status = message.get("status")
            self.document_section.doc_model.updateDocument(
                doc_id, status, message.get("progress", 0)
            )
            
        elif msg_type == "notification":
            self.showNotification(message.get("text", ""), message.get("level", "info"))
            
    def onPreferenceChanged(self, key: str, value: Any):
        """Handle preference changed"""
        if key == "theme":
            self.applyTheme()
            
    # ==================== ACTIONS ====================
    def showSettings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
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
        
                
    def toggleFullScreen(self):
        """Toggle full screen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
            
    def trayIconActivated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
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
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode()))
            
        # Window state
        state = self.prefs_manager.get("window_state")
        if state:
            self.restoreState(QByteArray.fromBase64(state.encode()))
            
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
        for viewer in self.document_section.active_viewers:
            viewer.close()
            
        # Cleanup
        self.websocket_client.disconnect()
        self.processing_worker.quit()
        self.processing_worker.wait()
        
        # Ask confirmation
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


# ==================== MAIN ====================
def main():
    # Create application
    app = LegalAIApplication(sys.argv)
    
    # Create splash screen
    splash = QSplashScreen()
    splash.setPixmap(QPixmap(600, 400))
    splash.showMessage(
        "Loading Legal AI System...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        Qt.GlobalColor.white
    )
    splash.show()
    
    # Process events
    app.processEvents()
    
    # Create main window
    window = IntegratedMainWindow()
    
    # Show window
    QTimer.singleShot(2000, splash.close)
    QTimer.singleShot(2000, window.show)
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
