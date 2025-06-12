import React, { useEffect } from 'react';
import useLoadingState from '../hooks/useLoadingState';
import { ViolationEntry } from '../types/violation';
import { fetchViolations, updateViolationStatus } from '../api/client';

const ReviewQueue: React.FC = () => {
  const { isLoading, error, data, executeAsync, setData } =
    useLoadingState<ViolationEntry[]>();

  useEffect(() => {
    executeAsync(() => fetchViolations());
  }, []);

  const handleResolve = async (id: string) => {
    try {
      await updateViolationStatus(id, 'RESOLVED');
      setData(data?.filter((v) => v.id !== id) || null);
    } catch {
      // ignore error for now
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Violations</h2>
      {isLoading && <div>Loading...</div>}
      {error && <div className="text-red-600">Failed to load reviews</div>}
      {data &&
        data.map((v) => (
          <div key={v.id} className="border p-4 rounded">
            <div className="font-medium mb-2">
              {v.violation_type} - {v.severity} ({v.confidence.toFixed(2)})
            </div>
            <p className="text-sm mb-2">{v.description}</p>
            <div className="mt-2 flex gap-2">
              <button
                className="px-2 py-1 bg-blue-600 text-white rounded"
                onClick={() => handleResolve(v.id)}
              >
                Mark Resolved
              </button>
            </div>
          </div>
        ))}
    </div>
  );
};

export default ReviewQueue;
