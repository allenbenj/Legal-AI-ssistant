from __future__ import annotations

"""Enhanced PyQt6 main window for the Legal AI desktop app."""

from PyQt6 import QtGui, QtWidgets


class AdvancedMainWindow(QtWidgets.QMainWindow):
    """Main window with menus, toolbars and an MDI area."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Legal AI Advanced UI")
        self.resize(1000, 700)
        self.mdi_area = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi_area)
        self.status_bar = self.statusBar()
        self._build_actions()
        self._build_menus()
        self._build_toolbars()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_actions(self) -> None:
        self.open_action = QtGui.QAction("Open", self)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self._open_file)

        self.settings_action = QtGui.QAction("Settings", self)
        self.settings_action.setShortcut("Ctrl+,")
        self.settings_action.triggered.connect(self._open_settings)

        self.exit_action = QtGui.QAction("Exit", self)
        self.exit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)

        self.upload_action = QtGui.QAction("Upload Document", self)
        self.upload_action.setShortcut("Ctrl+U")
        self.upload_action.triggered.connect(self._upload_document)

        self.workflow_action = QtGui.QAction("Start Workflow", self)
        self.workflow_action.setShortcut("Ctrl+W")
        self.workflow_action.triggered.connect(self._start_workflow)

        self.analytics_action = QtGui.QAction("View Analytics", self)
        self.analytics_action.setShortcut("Ctrl+A")
        self.analytics_action.triggered.connect(self._view_analytics)

    def _build_menus(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

    def _build_toolbars(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction(self.upload_action)
        toolbar.addAction(self.workflow_action)
        toolbar.addAction(self.analytics_action)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    def _open_file(self) -> None:  # pragma: no cover - GUI
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Document")
        if path:
            self.status_bar.showMessage(f"Opened: {path}", 3000)
            sub = QtWidgets.QMdiSubWindow()
            editor = QtWidgets.QTextEdit()
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    editor.setPlainText(fh.read())
            except Exception as exc:  # pragma: no cover - file errors
                editor.setPlainText(str(exc))
            sub.setWidget(editor)
            sub.setWindowTitle(path)
            self.mdi_area.addSubWindow(sub)
            sub.show()

    def _open_settings(self) -> None:  # pragma: no cover - GUI
        self.status_bar.showMessage("Opening settings...", 2000)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setLayout(QtWidgets.QVBoxLayout())
        dlg.layout().addWidget(QtWidgets.QLabel("Settings go here"))
        dlg.exec()

    def _upload_document(self) -> None:  # pragma: no cover - GUI
        self.status_bar.showMessage("Uploading document...", 2000)

    def _start_workflow(self) -> None:  # pragma: no cover - GUI
        self.status_bar.showMessage("Starting workflow...", 2000)

    def _view_analytics(self) -> None:  # pragma: no cover - GUI
        self.status_bar.showMessage("Opening analytics...", 2000)


__all__ = ["AdvancedMainWindow"]
