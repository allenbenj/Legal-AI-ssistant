import React from 'react';
import ReactDOM from 'react-dom/client';
import LegalAISystem from '@/legal-ai-gui';
import { AuthProvider } from './contexts/AuthContext';
import StatusDashboard from './components/StatusDashboard';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);

const App = () => {
  const [clientId] = React.useState(
    () => 'dashboard-' + Math.random().toString(36).slice(2),
  );
  return (
    <AuthProvider>
      <StatusDashboard clientId={clientId} />
      <LegalAISystem />
    </AuthProvider>
  );
};

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
