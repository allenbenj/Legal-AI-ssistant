import React, { useEffect, useState } from 'react';
import { Card, Grid } from '../design-system';
import useRealtimeSystemStatus, { SystemStatus } from '../hooks/useRealtimeSystemStatus';

interface ServiceStatusMap {
  [name: string]: {
    status: string;
    details?: string;
  };
}

interface HealthResponse {
  overall_status: string;
  services_status: ServiceStatusMap;
  performance_metrics_summary: {
    cpu?: number;
    memory?: number;
    disk?: number;
  };
  timestamp: string;
}

interface ServicesOverview {
  services: {
    [name: string]: {
      state: string;
      config_key?: string;
    };
  };
  active_workflow_config: Record<string, any>;
  timestamp: string;
}

const SystemHealth: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [servicesInfo, setServicesInfo] = useState<ServicesOverview | null>(null);
  const [clientId] = useState(() => 'health-' + Math.random().toString(36).slice(2));
  const { status } = useRealtimeSystemStatus(clientId);

  useEffect(() => {
    fetch('/api/v1/system/health')
      .then((res) => res.json())
      .then((data: HealthResponse) => setHealth(data))
      .catch((err) => {
        if (process.env.NODE_ENV !== 'production') {
          console.error('Failed to load system health', err);
        }
      });
  }, []);

  useEffect(() => {
    fetch('/api/v1/services')
      .then(res => res.json())
      .then((data: ServicesOverview) => setServicesInfo(data))
      .catch(err => {
        if (process.env.NODE_ENV !== 'production') {
          console.error('Failed to load services info', err);
        }
      });
  }, []);

  const cpu = status?.cpu ?? health?.performance_metrics_summary.cpu ?? 0;
  const memory = status?.memory ?? health?.performance_metrics_summary.memory ?? 0;
  const services = health?.services_status ?? {};
  const totalServices = Object.keys(services).length;
  const healthyServices = Object.values(services).filter((s) => s.status === 'healthy').length;

  const cardStyle: React.CSSProperties = { textAlign: 'center' };
  const valueStyle: React.CSSProperties = { fontSize: '1.5rem', fontWeight: 500 };

  return (
    <Grid columns={3} gap="md">
      <Card style={cardStyle}>
        <div style={{ marginBottom: '0.5rem' }}>CPU Usage</div>
        <div style={valueStyle}>{cpu}%</div>
      </Card>
      <Card style={cardStyle}>
        <div style={{ marginBottom: '0.5rem' }}>Memory Usage</div>
        <div style={valueStyle}>{memory}%</div>
      </Card>
      <Card style={cardStyle}>
        <div style={{ marginBottom: '0.5rem' }}>Services Healthy</div>
        <div style={valueStyle}>{healthyServices}/{totalServices}</div>
      </Card>
      <Card style={{ gridColumn: '1 / span 3' }}>
        <div style={{ marginBottom: '0.5rem', fontWeight: 500 }}>Service States</div>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {Object.entries(servicesInfo?.services || {}).map(([name, info]) => (
            <li key={name} style={{ marginBottom: '0.25rem' }}>
              <strong>{name}</strong>: {info.state}
            </li>
          ))}
        </ul>
      </Card>
    </Grid>
  );
};

export default SystemHealth;
