# API Endpoints

This document lists key REST endpoints exposed by the FastAPI backend. Each endpoint delegates to the `LegalAIIntegrationService` which orchestrates backend services.

## Legal Reasoning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/reasoning/analogical` | `POST` | Run analogical reasoning between two cases. |
| `/api/v1/reasoning/statute/interpret` | `POST` | Interpret statutory language in context. |
| `/api/v1/reasoning/constitutional` | `POST` | Perform constitutional analysis on provided text. |
| `/api/v1/reasoning/case-outcome` | `POST` | Predict likely case outcomes based on facts. |

Each endpoint accepts a JSON payload and returns the structured result from the `LegalReasoningEngine`.
