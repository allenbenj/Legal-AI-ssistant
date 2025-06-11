import React, { useEffect } from 'react';
import useLoadingState from '../hooks/useLoadingState';
import { ReviewItem, ReviewDecisionRequest } from '../types/review';
import { fetchPendingReviews, submitReviewDecision } from '../api/client';

const ReviewQueue: React.FC = () => {
  const { isLoading, error, data, executeAsync, setData } =
    useLoadingState<ReviewItem[]>();

  useEffect(() => {
    executeAsync(() => fetchPendingReviews());
  }, []);

  const handleDecision = async (itemId: string, decision: string) => {
    const req: ReviewDecisionRequest = { item_id: itemId, decision };
    try {
      await submitReviewDecision(req);
      setData(data?.filter((i) => i.item_id !== itemId) || null);
    } catch {
      // ignore error for now
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Pending Reviews</h2>
      {isLoading && <div>Loading...</div>}
      {error && <div className="text-red-600">Failed to load reviews</div>}
      {data &&
        data.map((item) => (
          <div key={item.item_id} className="border p-4 rounded">
            <div className="font-medium mb-2">
              {item.item_type} ({item.confidence.toFixed(2)})
            </div>
            <pre className="text-sm bg-gray-100 p-2 rounded overflow-auto">
              {JSON.stringify(item.content, null, 2)}
            </pre>
            <div className="mt-2 flex gap-2">
              <button
                className="px-2 py-1 bg-green-600 text-white rounded"
                onClick={() => handleDecision(item.item_id, 'approved')}
              >
                Approve
              </button>
              <button
                className="px-2 py-1 bg-red-600 text-white rounded"
                onClick={() => handleDecision(item.item_id, 'rejected')}
              >
                Reject
              </button>
            </div>
          </div>
        ))}
    </div>
  );
};

export default ReviewQueue;
