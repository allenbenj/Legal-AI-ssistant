export interface ReviewItem {
  item_id: string;
  item_type: string;
  content: Record<string, any>;
  confidence: number;
  source_document_id: string;
  extraction_context?: Record<string, any> | null;
  review_status: string;
  review_priority: string;
  created_at: string;
  reviewed_at?: string | null;
  reviewer_id?: string | null;
  reviewer_notes?: string | null;
}

export interface ReviewDecisionRequest {
  item_id: string;
  decision: string;
  modified_content?: Record<string, any> | null;
  reviewer_notes?: string | null;
  confidence_override?: number | null;
}

export interface ReviewDecisionResponse {
  status: string;
  item_id: string;
}

export interface ReviewStats {
  status_counts: Record<string, number>;
  priority_counts_pending: Record<string, number>;
  pending_reviews_total: number;
  new_items_last_24h: number;
  auto_approve_thresh: number;
  review_thresh: number;
  reject_thresh: number;
  db_path: string;
}
