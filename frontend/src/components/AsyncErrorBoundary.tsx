import React from 'react';
import ErrorBoundary from './ErrorBoundary';

export const DashboardErrorBoundary: React.FC<{children: React.ReactNode}> = ({ children }) => (
  <ErrorBoundary level="dashboard">{children}</ErrorBoundary>
);

export const DocumentProcessingErrorBoundary: React.FC<{children: React.ReactNode}> = ({ children }) => (
  <ErrorBoundary level="document-processing">{children}</ErrorBoundary>
);

export default ErrorBoundary;
