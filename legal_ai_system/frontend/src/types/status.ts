export interface SystemPerformance {
  cpu: number;
  memory: number;
  disk: number;
}

export interface ServiceHealth {
  total: number;
  healthy: number;
  unhealthy: number;
}

export interface DocumentMetrics {
  processing: number;
  completed: number;
  failed: number;
}

export interface SystemStatus {
  services: ServiceHealth;
  documents: DocumentMetrics;
  performance: SystemPerformance;
}

export enum AgentStatus {
  IDLE = "idle",
  PROCESSING = "processing",
  COMPLETED = "completed",
  ERROR = "error",
  CANCELLED = "cancelled",
}

export enum DocumentStatus {
  PENDING = "pending",
  LOADED = "loaded",
  PREPROCESSING = "preprocessing",
  PROCESSING = "processing",
  IN_REVIEW = "in_review",
  COMPLETED = "completed",
  ERROR = "error",
  CANCELLED = "cancelled",
}
