from __future__ import annotations

"""Reusable QAbstractTableModel implementations for REST data."""

from typing import Any, List

import pandas as pd
import requests
from PyQt6 import QtCore

from ..main_gui import APIClient


class _BaseRESTTableModel(QtCore.QAbstractTableModel):
    """Base model that loads table rows from a REST endpoint."""

    def __init__(self, api_client: APIClient, endpoint: str) -> None:
        super().__init__()
        self.api_client = api_client
        self.endpoint = endpoint
        self._df = pd.DataFrame()
        self._filtered_df = self._df

    # ------------------------------------------------------------------
    # Data loading and filtering
    # ------------------------------------------------------------------
    def fetch(self) -> None:
        """Load data from the configured REST endpoint."""
        url = f"{self.api_client.base_url}{self.endpoint}"
        try:
            resp = self.api_client.session.get(url)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                data = (
                    data.get("items")
                    or data.get("documents")
                    or data.get("violations")
                    or []
                )
            self._df = pd.DataFrame(data)
        except Exception:
            self._df = pd.DataFrame()
        self._filtered_df = self._df
        self.layoutChanged.emit()

    def filter_keyword(self, keyword: str) -> None:
        """Filter rows that contain ``keyword`` in any column."""
        if not keyword:
            self._filtered_df = self._df
        else:
            mask = self._df.apply(
                lambda r: r.astype(str).str.contains(keyword, case=False, na=False).any(),
                axis=1,
            )
            self._filtered_df = self._df[mask]
        self.layoutChanged.emit()

    # ------------------------------------------------------------------
    # Qt model overrides
    # ------------------------------------------------------------------
    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:  # type: ignore[override]
        return len(self._filtered_df.index)

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:  # type: ignore[override]
        return len(self._filtered_df.columns)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:  # type: ignore[override]
        if not index.isValid() or role not in (
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ):
            return None
        return str(self._filtered_df.iloc[index.row(), index.column()])

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:  # type: ignore[override]
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            cols: List[str] = list(self._filtered_df.columns)
            return cols[section] if section < len(cols) else None
        return str(section + 1)

    def sort(self, column: int, order: QtCore.Qt.SortOrder = QtCore.Qt.SortOrder.AscendingOrder) -> None:  # type: ignore[override]
        cols = list(self._filtered_df.columns)
        if 0 <= column < len(cols):
            ascending = order == QtCore.Qt.SortOrder.AscendingOrder
            self._filtered_df.sort_values(
                by=cols[column], ascending=ascending, inplace=True, kind="mergesort"
            )
            self.layoutChanged.emit()


class DocumentTableModel(_BaseRESTTableModel):
    """Table model for the ``/api/v1/documents`` endpoint."""

    def __init__(self, api_client: APIClient) -> None:
        super().__init__(api_client, "/api/v1/documents")


class ViolationsTableModel(_BaseRESTTableModel):
    """Table model for the ``/api/v1/violations`` endpoint."""

    def __init__(self, api_client: APIClient) -> None:
        super().__init__(api_client, "/api/v1/violations")


__all__ = ["DocumentTableModel", "ViolationsTableModel"]
