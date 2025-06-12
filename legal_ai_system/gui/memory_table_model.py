from __future__ import annotations

"""Model and widget for displaying memory entries in a table."""

import asyncio
from typing import Any, List, Optional

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from ..core.unified_memory_manager import MemoryType, UnifiedMemoryManager
from ..services.memory_service import memory_manager_context


class MemoryTableModel(QtCore.QAbstractTableModel):
    """Table model backed by a pandas ``DataFrame``."""

    def __init__(
        self,
        session_id: str = "memory_brain",
        manager: Optional[UnifiedMemoryManager] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.session_id = session_id
        self.manager = manager
        self.df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load conversation entries from ``UnifiedMemoryManager``."""

        async def _load_async(manager: UnifiedMemoryManager) -> pd.DataFrame:
            entries = await manager.get_context_window(self.session_id)
            data: List[dict[str, Any]] = []
            for entry in entries:
                if entry.memory_type != MemoryType.CONTEXT_WINDOW:
                    continue
                value = entry.value
                if isinstance(value, dict):
                    value = value.get("content", value)
                data.append(
                    {
                        "id": entry.id,
                        "speaker": entry.metadata.get("speaker", ""),
                        "content": value,
                        "source": entry.metadata.get("source", ""),
                        "created_at": entry.created_at.isoformat(),
                    }
                )
            return pd.DataFrame(data)

        async def _run() -> pd.DataFrame:
            if self.manager is not None:
                return await _load_async(self.manager)
            async with memory_manager_context() as mgr:
                return await _load_async(mgr)

        try:
            self.df = asyncio.run(_run())
        except Exception:
            self.df = pd.DataFrame()
        self.layoutChanged.emit()

    # ------------------------------------------------------------------
    # Qt model overrides
    # ------------------------------------------------------------------
    def rowCount(
        self, parent: QtCore.QModelIndex = QtCore.QModelIndex()
    ) -> int:  # noqa: D401,E501
        return 0 if parent.isValid() else len(self.df)

    def columnCount(
        self, parent: QtCore.QModelIndex = QtCore.QModelIndex()
    ) -> int:  # noqa: D401,E501
        return 0 if parent.isValid() else len(self.df.columns)

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        value = self.df.iat[index.row(), index.column()]
        return str(value)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            try:
                return str(self.df.columns[section])
            except IndexError:
                return None
        return str(section)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_dataframe(self) -> pd.DataFrame:
        """Return the underlying pandas DataFrame."""

        return self.df.copy()


class MemoryTableWidget(QtWidgets.QWidget):
    """Widget combining :class:`MemoryTableModel` with filtering and export."""

    def __init__(
        self,
        session_id: str = "memory_brain",
        manager: Optional[UnifiedMemoryManager] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.model = MemoryTableModel(session_id=session_id, manager=manager)
        self.model.load()

        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)  # search all columns
        self.proxy.setSourceModel(self.model)

        self.view = QtWidgets.QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter...")
        self.filter_edit.textChanged.connect(
            lambda text: self.proxy.setFilterRegularExpression(text)
        )

        export_csv_btn = QtWidgets.QPushButton("Export CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        export_excel_btn = QtWidgets.QPushButton("Export Excel")
        export_excel_btn.clicked.connect(self.export_excel)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.model.load)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.filter_edit)
        top.addWidget(refresh_btn)
        top.addWidget(export_csv_btn)
        top.addWidget(export_excel_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.view)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", filter="CSV Files (*.csv)"
        )
        if path:
            self.model.get_dataframe().to_csv(path, index=False)

    def export_excel(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Excel", filter="Excel Files (*.xlsx)"
        )
        if path:
            self.model.get_dataframe().to_excel(path, index=False)
