import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import types

# Stub heavy optional dependencies for tests
pydantic_mod = types.ModuleType("pydantic")

class BaseModel:
    def __init__(self, **data: object) -> None:
        for k, v in data.items():
            setattr(self, k, v)

pydantic_mod.BaseModel = BaseModel
sys.modules.setdefault("pydantic", pydantic_mod)

typer_mod = types.ModuleType("typer")

class Typer:
    def __init__(self, **kwargs: object) -> None:
        self.commands = {}

    def command(self, *args: object, **kwargs: object):
        def decorator(func):
            self.commands[func.__name__] = func
            return func
        return decorator

    def __call__(self, *args: object, **kwargs: object) -> None:
        pass


def Argument(default=None, help=""):
    return default


def echo(text: str) -> None:
    print(text)

typer_mod.Typer = Typer
typer_mod.Argument = Argument
typer_mod.echo = echo

testing_mod = types.ModuleType("typer.testing")

class CliRunner:
    def invoke(self, app, args):
        try:
            # use the first registered command
            cmd = next(iter(app.commands.values()))
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = cmd(args[0], args[1:])
            stdout = buf.getvalue() or str(result)
            return types.SimpleNamespace(exit_code=0, stdout=stdout)
        except Exception as e:
            return types.SimpleNamespace(exit_code=1, stdout="", exception=e)

testing_mod.CliRunner = CliRunner
sys.modules.setdefault("typer", typer_mod)
sys.modules.setdefault("typer.testing", testing_mod)

# Provide lightweight ServiceContainer for tests
service_container_mod = types.ModuleType("legal_ai_system.services.service_container")

class ServiceContainer:
    def __init__(self):
        self.registry = {}
        self._initialization_order = []
        self._service_states = {}

    async def register_service(self, name: str, factory):
        self.registry[name] = factory
        self._service_states[name] = types.SimpleNamespace(name="REGISTERED")

    async def initialize_all_services(self):
        for name in self.registry:
            self._initialization_order.append(name)
            self._service_states[name].name = "INITIALIZED"

service_container_mod.ServiceContainer = ServiceContainer
class _DummyLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        pass

import importlib.machinery
service_container_mod.__spec__ = importlib.machinery.ModuleSpec(
    "legal_ai_system.services.service_container",
    _DummyLoader(),
)
sys.modules.setdefault("legal_ai_system.services.service_container", service_container_mod)

# Minimal ServiceContainer stub used across tests

integration_pkg = types.ModuleType("legal_ai_system.integration_ready")
vector_store_mod = types.ModuleType("legal_ai_system.integration_ready.vector_store_enhanced")
vector_store_mod.MemoryStore = object
sys.modules.setdefault("legal_ai_system.integration_ready", integration_pkg)
sys.modules.setdefault("legal_ai_system.integration_ready.vector_store_enhanced", vector_store_mod)
