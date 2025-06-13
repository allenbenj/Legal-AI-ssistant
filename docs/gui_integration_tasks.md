# GUI Integration Tasks

This document lists the tasks required to connect the agents under
`legal_ai_system/agents` with the PyQt6 GUI application
`legal_ai_system/gui/legal-ai-pyqt6-enhanced.py`.

## Overview
The current GUI relies on stubbed network and database modules.
Agents and workflows are registered through the asynchronous
`ServiceContainer`, but the GUI does not instantiate or use the
container. Complete integration requires bridging the GUI with the
container so that actions performed in the interface invoke the
appropriate services and agents.

## Task List

1. **Create a Backend Bridge**
   - Implement a `BackendBridge` module in `legal_ai_system/gui`.
   - The bridge should initialise the `ServiceContainer` using
     `create_service_container()` and obtain a
     `LegalAIIntegrationService` instance.
   - Provide async helpers for document upload, status queries and
     text analysis that call the integration service.

2. **Replace Networking Stubs**
   - Replace `legal_ai_system/legal_ai_network/__init__.py` with an
     asynchronous API client that forwards requests from the GUI to the
     backend through the `BackendBridge`.
   - Remove the `AGENT_STUB` marker once the implementation is
     complete and update `docs/stub_backlog.md`.

3. **Implement Database Layer**
   - Expand `legal_ai_system/legal_ai_database/__init__.py` to
     persist uploaded documents and preferences.
   - Ensure methods used by the GUI emit appropriate Qt signals and
     interact with the persistence services from the service container.

4. **Wire GUI Actions**
   - Update `legal-ai-pyqt6-enhanced.py` so that the `uploadDocuments`
     and `processQueue` methods call the backend bridge instead of the
     current stubs.
   - Connect progress callbacks to update the dashboard and queue
     widgets in real time.

5. **Initialize Services on Application Start**
   - During `IntegratedMainWindow` construction, start the
     `BackendBridge` and wait for the service container to finish
     initialising before enabling UI actions.

6. **End-to-End Tests**
   - Add `nose2` integration tests that launch a minimal instance of
     the GUI and verify that document uploads trigger the expected
     service calls.
   - Include cleanup routines to tear down the event loop and close the
     service container after each test.

7. **Documentation and Examples**
   - Provide usage examples and update the README to describe how the
     GUI communicates with the backend services.

Completion of these tasks will ensure that all agents work in sequence
with the front-end GUI and that the supporting infrastructure is in
place.
