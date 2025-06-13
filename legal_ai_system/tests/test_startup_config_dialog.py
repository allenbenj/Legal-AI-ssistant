import os
import tempfile
import unittest

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from legal_ai_system.gui.startup_config_dialog import StartupConfigDialog


class StartupConfigDialogTest(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
        self.temp_dir = tempfile.TemporaryDirectory()
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, self.temp_dir.name)

    def tearDown(self):
        self.app.quit()
        self.temp_dir.cleanup()

    def test_save_and_env_vars(self):
        dlg = StartupConfigDialog()
        dlg.db_type_combo.setCurrentText("PostgreSQL")
        dlg.host_edit.setText("db.example.com")
        dlg.port_edit.setText("5432")
        dlg.user_edit.setText("user")
        dlg.password_edit.setText("pass")
        dlg.db_name_edit.setText("legal")
        dlg.llm_provider_combo.setCurrentText("openai")
        dlg.api_key_edit.setText("key123")
        dlg.save_settings()
        dlg.accept()
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "key123")
        self.assertEqual(os.environ.get("LLM_PROVIDER"), "openai")
        dlg2 = StartupConfigDialog()
        self.assertEqual(dlg2.llm_provider_combo.currentText(), "openai")
        self.assertEqual(dlg2.api_key_edit.text(), "key123")


if __name__ == "__main__":
    unittest.main()
