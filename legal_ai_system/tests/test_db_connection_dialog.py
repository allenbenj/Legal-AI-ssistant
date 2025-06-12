import os
import tempfile
import unittest

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from legal_ai_system.gui.db_connection_dialog import DBConnectionDialog


class DBConnectionDialogTest(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
        self.temp_dir = tempfile.TemporaryDirectory()
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, self.temp_dir.name)

    def tearDown(self):
        self.app.quit()
        self.temp_dir.cleanup()

    def test_save_and_load(self):
        dlg = DBConnectionDialog()
        dlg.db_type_combo.setCurrentText("PostgreSQL")
        dlg.host_edit.setText("example.com")
        dlg.port_edit.setText("5432")
        dlg.user_edit.setText("user")
        dlg.password_edit.setText("pass")
        dlg.db_name_edit.setText("legal")
        dlg.save_settings()
        del dlg
        dlg2 = DBConnectionDialog()
        self.assertEqual(dlg2.db_type_combo.currentText(), "PostgreSQL")
        self.assertEqual(dlg2.host_edit.text(), "example.com")
        self.assertEqual(dlg2.port_edit.text(), "5432")
        self.assertEqual(dlg2.user_edit.text(), "user")
        self.assertEqual(dlg2.password_edit.text(), "pass")
        self.assertEqual(dlg2.db_name_edit.text(), "legal")


if __name__ == "__main__":
    unittest.main()
