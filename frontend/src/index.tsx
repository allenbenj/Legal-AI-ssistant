import React from 'react';
import ReactDOM from 'react-dom/client';
import LegalAISystem from '@/legal-ai-gui';
import { AuthProvider } from './contexts/AuthContext';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <AuthProvider>
      <LegalAISystem />
    </AuthProvider>
  </React.StrictMode>
);
