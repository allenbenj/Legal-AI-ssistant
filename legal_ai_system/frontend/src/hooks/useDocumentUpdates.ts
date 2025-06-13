import { useEffect, useRef } from 'react';
import useWebSocket, { subscribe, unsubscribe } from './useWebSocket';

export interface DocumentUpdate {
  document_id: number | string;
  type: string;
  progress?: number;
  error?: string;
  [key: string]: any;
}

export default function useDocumentUpdates(
  clientId: string,
  documentIds: Array<number | string>,
  onUpdate: (update: DocumentUpdate) => void,
  onAgentUpdate?: (update: any) => void,
) {
  const { connected, send } = useWebSocket(`/ws/${clientId}`, handleMessage);
  const subscribed = useRef(false);

  function handleMessage(data: any) {
    if (data.type?.startsWith('processing_') || data.type === 'document_update') {
      onUpdate(data as DocumentUpdate);
    } else if (data.type?.startsWith('agent_') || data.type === 'agent_update') {
      onAgentUpdate?.(data);
    }
  }

  useEffect(() => {
    if (connected && !subscribed.current) {
      documentIds.forEach(id => subscribe({ send }, `document_updates_${id}`));
      subscribe({ send }, 'agent_updates');
      subscribed.current = true;
    }
    return () => {
      if (subscribed.current) {
        documentIds.forEach(id => unsubscribe({ send }, `document_updates_${id}`));
        unsubscribe({ send }, 'agent_updates');
        subscribed.current = false;
      }
    };
  }, [connected, documentIds.join('|')]);

  return { connected };
}
