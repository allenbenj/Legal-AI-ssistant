import React, { useState } from 'react';
import { Card, Grid } from '../design-system';
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


  const items = Object.values(workflows);

  return (
    <div style={{ marginBottom: spacing['2xl'] }}>
                  <div
                    style={{
                      width: '100%',
                      backgroundColor: colors.gray200,
                      height: '0.5rem',
                      borderRadius: spacing.xs,
                    }}
    </div>
  );
};

export default StatusDashboard;
