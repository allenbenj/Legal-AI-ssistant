# legal_ai_integrated.py - Complete integrated Legal AI Desktop Application

import asyncio
import json
import os
import sys

# Consolidated widget and helper classes from the former prototype packages
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

# Import all our custom modules
from legal_ai_system.gui.widgets.legal_ai_charts import (
    AnalyticsDashboardWidget,
    BarChartWidget,
    ChartData,
)
from legal_ai_system.legal_ai_database import (
    CacheManager,
    DatabaseManager,
    DocumentSearchEngine,
    PreferencesManager,
)
from legal_ai_system.legal_ai_network import (
    AsyncAPIClient,
    DocumentProcessingWorker,
    LegalAIAPIClient,
    NetworkManager,
    WebSocketClient,
)

from .backend_bridge import BackendBridge
from .startup_config_dialog import StartupConfigDialog


class GlowingButton(QPushButton):
    """Button that toggles a simple glow effect."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._glowing = False

    def startGlow(self) -> None:
        self._glowing = True
        self.setStyleSheet("background-color: #ffa500;")

    def stopGlow(self) -> None:
        self._glowing = False
        self.setStyleSheet("")


class FlipCard(QWidget):
    """Widget showing front/back text when clicked."""

    def __init__(self, front: str, back: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.front = QLabel(front)
        self.back = QLabel(back)
        self.back.hide()
        layout = QVBoxLayout(self)
        layout.addWidget(self.front)
        layout.addWidget(self.back)
        self.front.mousePressEvent = self._toggle  # type: ignore[assignment]
        self.back.mousePressEvent = self._toggle  # type: ignore[assignment]

    def _toggle(self, event) -> None:  # pragma: no cover - UI method
        if self.front.isVisible():
            self.front.hide()
            self.back.show()
        else:
            self.back.hide()
            self.front.show()


class TagCloud(QWidget):  # pragma: no cover - demo widget
    """Very small placeholder tag cloud."""

    def __init__(
        self, tags: list[str] | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.labels: list[QLabel] = []
        for tag in tags or []:
            lbl = QLabel(tag)
            self.labels.append(lbl)
            layout.addWidget(lbl)


class TimelineWidget(QWidget):  # pragma: no cover - demo widget
    """Placeholder timeline view."""


class NotificationWidget(QMessageBox):
    """Simple pop-up notification."""

    def __init__(
        self, message: str, level: str = "info", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setText(message)
        if level == "success":
            self.setIcon(QMessageBox.Icon.Information)
        elif level == "error":
            self.setIcon(QMessageBox.Icon.Critical)
        else:
            self.setIcon(QMessageBox.Icon.Warning)

    def show(self, parent: QWidget | None = None) -> None:  # type: ignore[override]
        if parent:
            super().show()
        else:
            super().exec()


class SearchableComboBox(QComboBox):  # pragma: no cover - minimal behaviour
    """Combo box with typing filter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().textEdited.connect(self._filter_items)

    def _filter_items(self, text: str) -> None:
        for i in range(self.count()):
            self.setRowHidden(i, text.lower() not in self.itemText(i).lower())


class DockablePanel(QDockWidget):  # pragma: no cover - stub
    """Simple dock widget."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)


@dataclass
class Document:
    """Simple document structure used by the GUI."""

    id: str
    filename: str
    status: str = "pending"
    progress: float = 0.0
    uploaded_at: datetime | None = None
    file_size: int = 0
    doc_type: str = "Unknown"


class DocumentTableModel(QAbstractTableModel):
    """Very small table model for storing documents."""

    headers = [
        "ID",
        "Filename",
        "Status",
        "Progress",
        "UploadedAt",
        "Size",
        "Type",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.documents = pd.DataFrame(columns=self.headers)

    # Qt model implementation -------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: D401
        return 0 if parent.isValid() else len(self.documents)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: D401
        return 0 if parent.isValid() else len(self.documents.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self.documents.iat[index.row(), index.column()]
        return str(value)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self.documents.columns[section])
            except IndexError:
                return None
        return str(section)

    # Convenience helpers ----------------------------------------------------
    def addDocument(self, doc: Document) -> None:
        row = {
            "ID": doc.id,
            "Filename": doc.filename,
            "Status": doc.status,
            "Progress": doc.progress,
            "UploadedAt": doc.uploaded_at.isoformat() if doc.uploaded_at else "",
            "Size": doc.file_size,
            "Type": doc.doc_type,
        }
        self.beginInsertRows(QModelIndex(), len(self.documents), len(self.documents))
        self.documents = pd.concat(
            [self.documents, pd.DataFrame([row])], ignore_index=True
        )
        self.endInsertRows()

    def updateDocument(self, doc_id: str, status: str, progress: float) -> None:
        idx = self.documents.index[self.documents["ID"] == doc_id]
        if not idx.empty:
            row = idx[0]
            self.documents.at[row, "Status"] = status
            self.documents.at[row, "Progress"] = progress
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, self.columnCount() - 1)
            )


class DocumentViewer(QWidget):
    """Trivial document viewer."""

    def __init__(self, doc_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Document {doc_id}")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Viewing document: {doc_id}"))


class SettingsDialog(QDialog):
    """Application preferences dialog."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.api_url_edit = QLineEdit()
        form.addRow("API URL", self.api_url_edit)

        self.openai_key_edit = QLineEdit()
        self.openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("OpenAI Key", self.openai_key_edit)

        self.regex_cb = QCheckBox("Regex Extraction")
        self.spacy_cb = QCheckBox("spaCy NER")
        self.bert_cb = QCheckBox("Legal-BERT")
        self.llm_cb = QCheckBox("LLM Extraction")

        form.addRow(self.regex_cb)
        form.addRow(self.spacy_cb)
        form.addRow(self.bert_cb)
        form.addRow(self.llm_cb)

        layout.addLayout(form)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)


class AboutDialog(QDialog):
    """Simple about dialog."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Legal AI System"))


class LegalAIApplication(QApplication):
    """Thin wrapper around :class:`QApplication`."""


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

        # Backend bridge and network client
        self.backend_bridge = BackendBridge()
        self.network_manager = NetworkManager(self.backend_bridge)
        self.api_client = LegalAIAPIClient(self.backend_bridge)
        self.processing_worker = DocumentProcessingWorker(self.api_client)

        # WebSocket placeholder for future real-time updates
        ws_url = self.prefs_manager.get("websocket_url", "ws://localhost:8000/ws")
        self.websocket_client = WebSocketClient(ws_url)

        # Document storage
        self.documents: Dict[str, Document] = {}
        self.active_viewers: List[DocumentViewer] = []

    def setupUI(self):
        """Setup main UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top toolbar with quick actions
        top_toolbar = QWidget()
        top_layout = QHBoxLayout(top_toolbar)
        top_layout.setContentsMargins(10, 5, 10, 5)

        # Search bar
        self.global_search = QLineEdit()
        self.global_search.setPlaceholderText(
            "Search documents, entities, or legal terms..."
        )
        self.global_search.returnPressed.connect(self.performGlobalSearch)
        top_layout.addWidget(self.global_search)

        # Quick action buttons
        self.upload_btn = GlowingButton("Upload")
        self.upload_btn.clicked.connect(self.uploadDocuments)
        self.upload_btn.setEnabled(False)
        top_layout.addWidget(self.upload_btn)

        self.process_btn = GlowingButton("Process Queue")
        self.process_btn.clicked.connect(self.processQueue)
        self.process_btn.setEnabled(False)
        top_layout.addWidget(self.process_btn)

        main_layout.addWidget(top_toolbar)

        # Main content area with tabs
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
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

    def createDashboard(self) -> QWidget:
        """Create dashboard with overview widgets"""
        dashboard = QWidget()
        layout = QGridLayout(dashboard)

        # Stats cards
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.Box)
        stats_layout = QHBoxLayout(stats_frame)

        # Create flip cards for stats
        self.doc_count_card = FlipCard("Total Documents\n0", "Click for details")
        stats_layout.addWidget(self.doc_count_card)

        self.success_rate_card = FlipCard("Success Rate\n0%", "Processing accuracy")
        stats_layout.addWidget(self.success_rate_card)

        self.active_users_card = FlipCard("Active Users\n0", "Currently online")
        stats_layout.addWidget(self.active_users_card)

        layout.addWidget(stats_frame, 0, 0, 1, 2)

        # Recent activity timeline
        self.timeline = TimelineWidget()
        timeline_frame = QGroupBox("Recent Activity")
        timeline_layout = QVBoxLayout(timeline_frame)
        timeline_layout.addWidget(self.timeline)
        layout.addWidget(timeline_frame, 1, 0)

        # Tag cloud
        self.tag_cloud = TagCloud()
        tag_frame = QGroupBox("Popular Tags")
        tag_layout = QVBoxLayout(tag_frame)
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

    def createDocumentsView(self) -> QWidget:
        """Create documents management view"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)

        # Filter controls
        toolbar_layout.addWidget(QLabel("Status:"))
        self.status_filter = SearchableComboBox()
        self.status_filter.addItems(
            ["All", "Pending", "Processing", "Completed", "Failed"]
        )
        self.status_filter.currentTextChanged.connect(self.filterDocuments)
        toolbar_layout.addWidget(self.status_filter)

        toolbar_layout.addWidget(QLabel("Type:"))
        self.type_filter = SearchableComboBox()
        self.type_filter.addItems(
            ["All", "Contract", "Legal Brief", "Patent", "Compliance"]
        )
        toolbar_layout.addWidget(self.type_filter)

        toolbar_layout.addStretch()

        # Export button
        export_btn = QPushButton("Export Selected")
        export_btn.clicked.connect(self.exportDocuments)
        toolbar_layout.addWidget(export_btn)

        layout.addWidget(toolbar)

        # Document table
        self.doc_table = QTableView()
        self.doc_model = DocumentTableModel()
        self.doc_table.setModel(self.doc_model)
        self.doc_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.doc_table.setAlternatingRowColors(True)
        self.doc_table.doubleClicked.connect(self.viewDocument)

        # Context menu
        self.doc_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.doc_table.customContextMenuRequested.connect(self.showDocumentContextMenu)

        layout.addWidget(self.doc_table)

        return widget

    def createQueueView(self) -> QWidget:
        """Create processing queue view"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Queue controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)

        pause_btn = QPushButton("Pause Queue")
        pause_btn.setCheckable(True)
        pause_btn.toggled.connect(self.toggleQueueProcessing)
        controls_layout.addWidget(pause_btn)

        clear_btn = QPushButton("Clear Completed")
        clear_btn.clicked.connect(self.clearCompletedItems)
        controls_layout.addWidget(clear_btn)

        controls_layout.addStretch()

        layout.addWidget(controls)

        # Queue list
        self.queue_list = QListWidget()
        self.queue_list.setAlternatingRowColors(True)
        layout.addWidget(self.queue_list)

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

        db_conn_action = tools_menu.addAction("Database Connections...")
        db_conn_action.triggered.connect(self.showDatabaseConnections)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        docs_action = help_menu.addAction("Documentation")
        docs_action.setShortcut("F1")
        docs_action.triggered.connect(self.openDocumentation)

        dashboard_action = help_menu.addAction("Open Dashboard")
        dashboard_action.triggered.connect(self.openDashboard)

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
        self.network_manager.connectionStatusChanged.connect(
            self.onConnectionStatusChanged
        )
        self.api_client.documentsLoaded.connect(self.onDocumentsLoaded)
        self.api_client.documentUploaded.connect(self.onDocumentUploaded)
        self.api_client.processingProgress.connect(self.onProcessingProgress)

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
            self.setStyleSheet(
                """
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
            """
            )

    def startServices(self):
        """Start background services"""
        self.log("Starting services...")

        # Start backend bridge
        self.backend_bridge.serviceReady.connect(self.onBackendReady)
        self.backend_bridge.start()

        # Connect WebSocket placeholder
        self.websocket_client.connect()

    # ==================== SLOTS ====================
    def onDatabaseReady(self):
        """Handle database ready"""
        self.log("Database initialized")
        self.loadLocalDocuments()

    def onDatabaseError(self, error: str):
        """Handle database error"""
        self.log(f"Database error: {error}", level="error")
        QMessageBox.critical(self, "Database Error", error)

    def onBackendReady(self):
        """Backend bridge initialised."""
        self.upload_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.network_manager.checkConnection()
        self.loadDocuments()
        self.updateDashboard()
        self.refreshAgentsStatus()

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
                doc_type=doc_data.get("type", "Unknown"),
            )
            self.documents[doc.id] = doc
            self.doc_model.addDocument(doc)

            # Save to local database
            self.db_manager.saveDocument(
                doc.id,
                doc.filename,
                file_size=doc.file_size,
                metadata=doc_data.get("metadata"),
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
                results,
            )

        self.showNotification(
            f"Processing complete: {self.documents[doc_id].filename}", "success"
        )

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
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents",
            str(Path.home()),
            "Documents (*.pdf *.docx *.txt *.md);;All Files (*.*)",
        )

        if files:
            self.upload_btn.startGlow()

            for file_path in files:
                path = Path(file_path)

                self.api_client.uploadDocument(
                    path,
                    {
                        "enable_regex_extraction": self.prefs_manager.get(
                            "enable_regex_extraction", True
                        ),
                        "enable_spacy_ner": self.prefs_manager.get(
                            "enable_spacy_ner", False
                        ),
                        "enable_legal_bert": self.prefs_manager.get(
                            "enable_legal_bert", False
                        ),
                        "enable_llm": self.prefs_manager.get("enable_llm", True),
                        "confidence_threshold": self.prefs_manager.get(
                            "confidence_threshold", 0.7
                        ),
                    },
                )

            self.upload_btn.stopGlow()

    def processQueue(self):
        """Process document queue"""
        self.process_btn.startGlow()

        for doc_id, doc in self.documents.items():
            if doc.status == "pending":
                self.api_client.uploadDocument(
                    Path(doc.filename),
                    {
                        "enable_regex_extraction": self.prefs_manager.get(
                            "enable_regex_extraction", True
                        ),
                        "enable_spacy_ner": self.prefs_manager.get(
                            "enable_spacy_ner", False
                        ),
                        "enable_legal_bert": self.prefs_manager.get(
                            "enable_legal_bert", False
                        ),
                        "enable_llm": self.prefs_manager.get("enable_llm", True),
                    },
                )

    def viewDocument(self, index: QModelIndex):
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
            results_widget = QListWidget()
            for result in results:
                item = QListWidgetItem(f"{result['filename']}: {result['snippet']}")
                item.setData(Qt.ItemDataRole.UserRole, result["document_id"])
                results_widget.addItem(item)

            self.main_tabs.addTab(results_widget, f"Search: {query}")
            self.main_tabs.setCurrentWidget(results_widget)

    def showSettings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        dialog.api_url_edit.setText(
            str(self.prefs_manager.get("api_url", "http://localhost:8000"))
        )
        dialog.openai_key_edit.setText(str(self.prefs_manager.get("openai_key", "")))
        dialog.regex_cb.setChecked(
            bool(self.prefs_manager.get("enable_regex_extraction", True))
        )
        dialog.spacy_cb.setChecked(
            bool(self.prefs_manager.get("enable_spacy_ner", False))
        )
        dialog.bert_cb.setChecked(
            bool(self.prefs_manager.get("enable_legal_bert", False))
        )
        dialog.llm_cb.setChecked(bool(self.prefs_manager.get("enable_llm", True)))

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Save settings
            self.prefs_manager.set("api_url", dialog.api_url_edit.text())
            self.prefs_manager.set("openai_key", dialog.openai_key_edit.text())
            self.prefs_manager.set(
                "enable_regex_extraction", dialog.regex_cb.isChecked()
            )
            self.prefs_manager.set("enable_spacy_ner", dialog.spacy_cb.isChecked())
            self.prefs_manager.set("enable_legal_bert", dialog.bert_cb.isChecked())
            self.prefs_manager.set("enable_llm", dialog.llm_cb.isChecked())

    def showDatabaseConnections(self):
        """Open database connection configuration dialog."""
        from .db_connection_dialog import DBConnectionDialog

        dialog = DBConnectionDialog(self)
        dialog.exec()

    def showAbout(self):
        """Show about dialog"""
        dialog = AboutDialog(self)
        dialog.exec()

    def openDocumentation(self):
        """Open project documentation in the default browser."""
        doc_path = Path(__file__).resolve().parents[2] / "README.md"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(doc_path)))

    def openDashboard(self):
        """Open the Streamlit dashboard if running."""
        QDesktopServices.openUrl(QUrl("http://localhost:8501"))

    def showNotification(self, message: str, notification_type: str = "info"):
        """Show notification popup"""
        notif = NotificationWidget(message, notification_type)
        notif.show(self)

    def log(self, message: str, level: str = "info"):
        """Log message to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if level == "error":
            html = (
                f'<span style="color: #f44336;">[{timestamp}] ERROR: {message}</span>'
            )
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
        now = QDateTime.currentDateTime()
        for i, (doc_id, doc) in enumerate(list(self.documents.items())[:5]):
            self.timeline.addEvent(
                now.addSecs(-i * 3600),
                doc.filename,
                f"Status: {doc.status}",
                "success" if doc.status == "completed" else "info",
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
                file_size=doc_data.get("file_size", 0),
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

    def showDocumentContextMenu(self, pos: QPoint):
        """Show context menu for documents"""
        menu = QMenu(self)

        view_action = menu.addAction("View")
        view_action.triggered.connect(
            lambda: self.viewDocument(self.doc_table.currentIndex())
        )

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
            "window_geometry", bytes(self.saveGeometry().toBase64()).decode()
        )

        # Window state
        self.prefs_manager.set(
            "window_state", bytes(self.saveState().toBase64()).decode()
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
        if hasattr(self, "agent_refresh_timer"):
            self.agent_refresh_timer.stop()

        # Ask confirmation
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


# ==================== MAIN ====================
def main():
    # Create application
    app = LegalAIApplication(sys.argv)

    # Display initial configuration dialog
    setup_dialog = StartupConfigDialog()
    if setup_dialog.exec() != QDialog.DialogCode.Accepted:
        return

    # Create splash screen
    splash = QSplashScreen()
    splash.setPixmap(QPixmap(600, 400))
    splash.showMessage(
        "Loading Legal AI System...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        Qt.GlobalColor.white,
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
