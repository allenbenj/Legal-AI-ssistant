import { useEffect, useState } from 'react';

export interface MetricsSnapshot {
  kg_queries_total: number;
  kg_query_cache_hits: number;
  vector_add_seconds_sum: number;
  vector_search_seconds_sum: number;
  pg_pool_in_use: number;
  pg_pool_free: number;
  redis_pool_in_use: number;
}

export default function useMetrics(interval = 5000) {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);

  useEffect(() => {
    let isMounted = true;

    const fetchMetrics = () => {
      fetch('/metrics')
        .then(res => res.json())
        .then(data => {
          if (isMounted) setMetrics(data as MetricsSnapshot);
        })
        .catch(() => {});
    };

    fetchMetrics();
    const id = setInterval(fetchMetrics, interval);
    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, [interval]);

  return metrics;
}
