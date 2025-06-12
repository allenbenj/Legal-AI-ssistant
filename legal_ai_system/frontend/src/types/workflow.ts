export interface ProcessingRequest {
  enable_ner: boolean;
  enable_llm_extraction: boolean;
  enable_confidence_calibration: boolean;
  confidence_threshold: number;
}

export interface WorkflowConfig extends ProcessingRequest {
  id: string;
  name: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface WorkflowConfigCreate extends ProcessingRequest {
  name: string;
  description?: string | null;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  size_bytes: number;
  status: string;
  message?: string | null;
}

export interface DocumentStatusResponse {
  document_id: string;
  status: string;
  progress: number;
  stage?: string | null;
  estimated_completion_sec?: number | null;
  result_summary?: Record<string, any> | null;
}
