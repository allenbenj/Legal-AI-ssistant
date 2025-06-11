import types
import importlib
import sys


def test_memory_brain_gui_main(mocker):
    st = types.SimpleNamespace(set_page_config=mocker.Mock())
    mocker.patch.dict(sys.modules, {"streamlit": st})
    panel_cls = mocker.Mock()
    mocker.patch.dict(
        sys.modules,
        {"legal_ai_system.gui.panels.memory_brain_panel": types.SimpleNamespace(MemoryBrainPanel=panel_cls)}
    )
    module = importlib.import_module("legal_ai_system.scripts.memory_brain_gui")
    module.main()
    st.set_page_config.assert_called_once_with(page_title="Memory Brain", page_icon="🧠", layout="wide")
    panel_cls.assert_called_once()
    panel_cls.return_value.render.assert_called_once()
