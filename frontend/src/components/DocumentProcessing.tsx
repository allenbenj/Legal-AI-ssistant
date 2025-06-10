import React, { useEffect, useState } from 'react';
import { DocumentStatus } from '../types';

interface DocumentProcessingProps {
  documentId: string;
  pollingIntervalMs?: number;
}

interface StatusResponse {
  status: DocumentStatus;
  progress?: number;
  stage?: string;
}

const statusClasses: Record<DocumentStatus, string> = {
  [DocumentStatus.PENDING]: 'bg-gray-100 text-gray-700',
  [DocumentStatus.LOADED]: 'bg-blue-100 text-blue-700',
  [DocumentStatus.PREPROCESSING]: 'bg-blue-100 text-blue-700',
  [DocumentStatus.PROCESSING]: 'bg-blue-100 text-blue-700',
  [DocumentStatus.IN_REVIEW]: 'bg-yellow-100 text-yellow-700',
  [DocumentStatus.COMPLETED]: 'bg-green-100 text-green-700',
  [DocumentStatus.ERROR]: 'bg-red-100 text-red-700',
  [DocumentStatus.CANCELLED]: 'bg-gray-200 text-gray-700',
};

const DocumentProcessing: React.FC<DocumentProcessingProps> = ({
  documentId,
  pollingIntervalMs = 5000,
}) => {
  const [status, setStatus] = useState<DocumentStatus | null>(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let active = true;

    async function fetchStatus() {
      try {
        const res = await fetch(`/api/v1/documents/${documentId}/status`);
        if (!res.ok) {
          return;
        }
        const data: StatusResponse = await res.json();
        if (active) {
          setStatus(data.status);
          setProgress(Math.round((data.progress ?? 0) * 100));
        }
      } catch {
        // network errors ignored; next poll will retry
      }
    }

    fetchStatus();
    const id = setInterval(fetchStatus, pollingIntervalMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [documentId, pollingIntervalMs]);

  const label = status ?? DocumentStatus.PENDING;
  const statusClass = status ? statusClasses[status] : statusClasses[DocumentStatus.PENDING];

  return (
    <div className="space-y-2">
      <div className={`inline-block px-2 py-1 rounded text-xs ${statusClass}`}>{label}</div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-600 h-2 rounded-full transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="text-sm text-gray-600">{progress}%</div>
    </div>
  );
};

export default DocumentProcessing;
