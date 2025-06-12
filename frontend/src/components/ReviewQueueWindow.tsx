import React, { useEffect } from 'react';
import useLoadingState from '../hooks/useLoadingState';
import { ViolationRecord, ViolationDecisionRequest } from '../types/violation';
import { fetchPendingViolations, submitViolationDecision } from '../api/client';

const ReviewQueueWindow: React.FC = () => {
  const { isLoading, error, data, executeAsync, setData } =
    useLoadingState<ViolationRecord[]>();

  useEffect(() => {
    executeAsync(() => fetchPendingViolations());
  }, []);

  const handleDecision = async (id: string, decision: string) => {
    const req: ViolationDecisionRequest = { decision };
    try {
      await submitViolationDecision(id, req);
      setData(data?.filter((v) => v.id !== id) || null);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Pending Violations</h2>
      {isLoading && <div>Loading...</div>}
      {error && <div className="text-red-600">Failed to load violations</div>}
      {data &&
        data.map((v) => (
          <div key={v.id} className="border p-4 rounded">
            <div className="font-medium mb-2">
              {v.violation_type} ({v.confidence.toFixed(2)})
            </div>
            <p className="text-sm mb-2">{v.description}</p>
            <div className="mt-2 flex gap-2">
              <button
                className="px-2 py-1 bg-green-600 text-white rounded"
                onClick={() => handleDecision(v.id, 'approved')}
              >
                Approve
              </button>
              <button
                className="px-2 py-1 bg-red-600 text-white rounded"
                onClick={() => handleDecision(v.id, 'rejected')}
              >
                Reject
              </button>
            </div>
          </div>
        ))}
    </div>
  );
};

export default ReviewQueueWindow;
