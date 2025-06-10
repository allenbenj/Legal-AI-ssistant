import { useEffect, useRef, useState } from 'react';

export interface WebSocketHook {
  connected: boolean;
  send: (data: any) => void;
}

export default function useWebSocket(url: string, onMessage: (msg: any) => void): WebSocketHook {
  const wsRef = useRef<WebSocket>();
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();
  const [connected, setConnected] = useState(false);

  const connect = () => {
    wsRef.current = new WebSocket(url);
    wsRef.current.onopen = () => {
      setConnected(true);
    };
    wsRef.current.onclose = () => {
      setConnected(false);
      reconnectRef.current = setTimeout(connect, 2000);
    };
    wsRef.current.onmessage = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data);
        onMessage(data);
      } catch {
        // ignore malformed messages
      }
    };
  };

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
    };
  }, [url]);

  const send = (data: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  };

  return { connected, send };
}

export const subscribe = (ws: { send: (data: any) => void }, topic: string) => {
  ws.send({ type: 'subscribe', topic });
};

export const unsubscribe = (ws: { send: (data: any) => void }, topic: string) => {
  ws.send({ type: 'unsubscribe', topic });
};
