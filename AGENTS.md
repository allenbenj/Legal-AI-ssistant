Instructions for Stub Creation and Comprehensive Implementation Workflow
Core Principle: Stubs are temporary placeholders to facilitate initial scaffolding and dependency resolution. They are a debt that must be repaid with full, robust implementation. Your workflow must explicitly account for and prioritize the completion of these stubs.

1. Clear Definition & Marking of Stubs
What is a Stub: A "stub" is defined as a piece of code (function, method, class, API endpoint, UI component) that:
Provides a minimal, temporary, or simulated output (e.g., returns dummy data, None, True, or logs a "not implemented" message).
Contains placeholder logic instead of the actual, complete business logic.
Is created to satisfy a dependency for another component or to allow initial integration/testing of a higher-level flow.
Mandatory Stub Markers: Every stub you create MUST be clearly marked in the code using standardized comments or annotations. Use a consistent pattern for easy identification:
# TODO: AGENT_STUB - [Brief description of what needs implementation]
# AGENT_STUB_PENDING: [Brief description and expected output/behavior]
For more complex stubs, consider adding a docstring marker:
Python

def get_user_profile(user_id):
    """
    AGENT_STUB_PENDING: Fully implement fetching user profile from DB.
    Currently returns dummy data.
    """
    # TODO: AGENT_STUB - Replace with actual DB query
    return {"id": user_id, "name": "Test User", "email": "test@example.com"}
Internal Stub Backlog: In addition to code markers, maintain an internal, dynamic list (or "Stub Backlog") of every stub created. This list should include:
Unique Identifier (e.g., file:line or function name).
Brief description of the stub's purpose.
The "real" functionality it needs to implement.
Severity/Priority (e.g., "Critical path," "Minor feature," "Error handling").
Dependent components (what other parts of the code rely on this stub).
2. Prioritization & Scheduling of Implementation
Initial Pass for Scaffolding: During initial task decomposition and coding, it is permissible to create stubs to rapidly build out the application's structure and connect high-level components.
Prioritize Critical Path: Once the basic structure is in place, immediately shift focus to implementing stubs that are on the critical path of the core application functionality. This includes:
APIs that will be called by essential frontend features.
Backend logic that supports core user stories.
Database interactions for primary data.
Dependency-Driven Implementation: Prioritize stubs that are blocking other components from being fully implemented or tested with live data.
Sequential Completion: Strive to complete stubs in a logical order, often from the "bottom-up" (data layer, then business logic, then API endpoints, then frontend consumption).
Dedicated "Stub Implementation" Phases: Allocate explicit time or "sprints" within your work plan for "Stub Implementation." Do not consider a task complete until all relevant stubs for that task are fully implemented.
3. Comprehensive Implementation Process
Replace Placeholder Logic: For each stub identified in your Stub Backlog:
Remove all dummy data and placeholder return values.
Implement the full, correct business logic as designed in your planning phase.
Integrate with actual dependencies (e.g., make real database queries, call external services, interact with other internal modules).
Add Robustness:
Implement comprehensive error handling for all potential failure points (network issues, invalid input, resource unavailability, unexpected external service responses).
Add informative logging at appropriate levels (DEBUG, INFO, ERROR) to aid in future debugging and monitoring.
Perform input validation where necessary.
Write Thorough Tests:
Develop unit tests that cover the logic of the newly implemented stub.
Develop integration tests that verify its interaction with its dependencies (e.g., database, other services).
Update or create end-to-end tests to ensure the full flow, now using live data, works correctly.
Remove Stub Markers: Once a stub is fully implemented and tested, immediately remove all AGENT_STUB markers from the code and remove it from your internal Stub Backlog. This signifies completion and prevents confusion.
4. Continuous Validation & Self-Correction
Daily Stub Check: At the start of each work cycle, review your Stub Backlog and scan the codebase for AGENT_STUB markers to ensure no stubs are forgotten or left behind.
Integration Check with Live Data: Before considering any feature "complete," run it against a live (or staging) backend with actual data. Explicitly verify that test data is no longer being used for core functionality.
Automated Linting/Checking: Implement a simulated linter or static analysis step that specifically flags any remaining AGENT_STUB markers in committed code. This should be a blocker for considering a feature "done."
Performance & Security Review: Once implemented, include the stub's completed code in performance and security assessments.
