// Centralized API client for backend REST endpoints
import {
  WorkflowConfig,
  WorkflowConfigCreate,
  DocumentUploadResponse,
  ProcessingRequest,
  DocumentStatusResponse,
} from "../types/workflow";
import {
  ReviewItem,
  ReviewDecisionRequest,
  ReviewDecisionResponse,
  ReviewStats,
} from "../types/review";
import { ViolationRecord, ViolationDecisionRequest } from "../types/violation";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function listWorkflows(): Promise<WorkflowConfig[]> {
  const res = await fetch("/api/v1/workflows");
  return handleResponse<WorkflowConfig[]>(res);
}

export async function createWorkflow(data: WorkflowConfigCreate): Promise<WorkflowConfig> {
  const res = await fetch("/api/v1/workflows", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  return handleResponse<WorkflowConfig>(res);
}

export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/v1/documents/upload", {
    method: "POST",
    body: form,
  });
  return handleResponse<DocumentUploadResponse>(res);
}

export async function processDocument(documentId: string, request: ProcessingRequest): Promise<void> {
  const res = await fetch(`/api/v1/documents/${documentId}/process`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
  await handleResponse(res);
}

export async function getDocumentStatus(documentId: string): Promise<DocumentStatusResponse> {
  const res = await fetch(`/api/v1/documents/${documentId}/status`);
  return handleResponse<DocumentStatusResponse>(res);
}

export async function fetchPendingReviews(limit = 20): Promise<ReviewItem[]> {
  const res = await fetch(`/api/v1/reviews/pending?limit=${limit}`);
  return handleResponse<ReviewItem[]>(res);
}

export async function submitReviewDecision(
  data: ReviewDecisionRequest
): Promise<ReviewDecisionResponse> {
  const res = await fetch(`/api/v1/reviews/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse<ReviewDecisionResponse>(res);
}

export async function fetchReviewStats(): Promise<ReviewStats> {
  const res = await fetch(`/api/v1/reviews/stats`);
  return handleResponse<ReviewStats>(res);
}

export async function fetchPendingViolations(): Promise<ViolationRecord[]> {
  const res = await fetch(`/api/v1/violations?status=pending`);
  return handleResponse<ViolationRecord[]>(res);
}

export async function submitViolationDecision(
  id: string,
  data: ViolationDecisionRequest
): Promise<void> {
  const res = await fetch(`/api/v1/violations/${id}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  await handleResponse(res);
}
