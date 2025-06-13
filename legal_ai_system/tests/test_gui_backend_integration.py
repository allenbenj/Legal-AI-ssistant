import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from legal_ai_system.gui.legal_ai_pyqt6_enhanced import IntegratedMainWindow


class GuiUploadIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    def tearDown(self):
        self.app.quit()

    def test_upload_invokes_backend(self):
        with patch("legal_ai_system.gui.backend_bridge.BackendBridge") as Bridge:
            bridge = Bridge.return_value
            bridge.start.return_value = None
            bridge.serviceReady = MagicMock()
            bridge.upload_document = MagicMock()

            window = IntegratedMainWindow()
            window.backend_bridge = bridge
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(b"data")
            try:
                with patch("PyQt6.QtWidgets.QFileDialog.getOpenFileNames", return_value=([tf.name], "")):
                    window.uploadDocuments()
                    bridge.upload_document.assert_called_once()
            finally:
                os.remove(tf.name)
            window.close()

