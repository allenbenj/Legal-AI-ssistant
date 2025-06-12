import { useEffect, useState, useRef } from 'react';
import useWebSocket, { subscribe, unsubscribe } from './useWebSocket';

export interface SystemStatus {
  cpu: number;
  memory: number;
  disk: number;
  timestamp: string;
}

export default function useRealtimeSystemStatus(clientId: string) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const { connected, send } = useWebSocket(`/ws/${clientId}`, handleMessage);
  const subscribed = useRef(false);

  function handleMessage(data: any) {
    if (data.type === 'system_status') {
      setStatus({
        cpu: data.cpu,
        memory: data.memory,
        disk: data.disk,
        timestamp: data.timestamp,
      });
    }
  }

  useEffect(() => {
    if (connected && !subscribed.current) {
      subscribe({ send }, 'system_status');
      subscribe({ send }, 'document_updates');
      subscribe({ send }, 'agent_updates');
      subscribed.current = true;
    }
    return () => {
      if (subscribed.current) {
        unsubscribe({ send }, 'system_status');
        unsubscribe({ send }, 'document_updates');
        unsubscribe({ send }, 'agent_updates');
        subscribed.current = false;
      }
    };
  }, [connected]);

  return { status, connected, send };
}
