import React, { useEffect, useRef, useState } from 'react';
import { Card, Grid } from '../design-system';
import useWebSocket, { subscribe, unsubscribe } from '../hooks/useWebSocket';
import { spacing, colors } from '../design-system/tokens';

interface WorkflowUpdate {
  workflow_id: string;
  progress: number;
  stage?: string;
}

export interface StatusDashboardProps {
  clientId: string;
}

const StatusDashboard: React.FC<StatusDashboardProps> = ({ clientId }) => {
  const [workflows, setWorkflows] = useState<Record<string, WorkflowUpdate>>({});
  const { connected, send } = useWebSocket(`/ws/${clientId}`, handleMessage);
  const subscribed = useRef(false);

  function handleMessage(data: any) {
    if (data.type === 'workflow_progress') {
      setWorkflows(prev => ({
        ...prev,
        [data.workflow_id]: {
          workflow_id: data.workflow_id,
          progress: data.progress ?? 0,
          stage: data.stage,
        },
      }));
    }
  }

  useEffect(() => {
    if (connected && !subscribed.current) {
      subscribe({ send }, 'workflow_updates');
      subscribed.current = true;
    }
    return () => {
      if (subscribed.current) {
        unsubscribe({ send }, 'workflow_updates');
        subscribed.current = false;
      }
    };
  }, [connected]);

  const items = Object.values(workflows);

  return (
    <div style={{ marginBottom: spacing['2xl'] }}>
      <h3 style={{ marginBottom: spacing.sm }}>Workflow Status</h3>
      {items.length === 0 ? (
        <div style={{ color: colors.gray600 }}>No workflows running</div>
      ) : (
        <Grid columns={1} gap="md">
          {items.map(item => {
            const pct = Math.round((item.progress ?? 0) * 100);
            return (
              <Card key={item.workflow_id} style={{ padding: spacing.sm }}>
                <div style={{ marginBottom: spacing.xs, fontWeight: 500 }}>
                  {item.stage || 'Processing'} ({pct}%)
                </div>
                <div
                  style={{
                    width: '100%',
                    backgroundColor: colors.gray200,
                    height: '0.5rem',
                    borderRadius: spacing.xs,
                  }}
                >
                  <div
                    style={{
                      width: `${pct}%`,
                      backgroundColor: colors.primary,
                      height: '100%',
                      borderRadius: spacing.xs,
                      transition: 'width 0.3s',
                    }}
                  />
                </div>
              </Card>
            );
          })}
        </Grid>
      )}
    </div>
  );
};

export default StatusDashboard;
