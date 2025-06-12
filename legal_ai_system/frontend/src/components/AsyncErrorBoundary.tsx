import React from 'react';
import ErrorBoundary from './ErrorBoundary';

const DefaultFallback = (
  reset: () => void,
  error: Error | null,
  label: string,
) => (
  <div className="p-8 text-center">
    <h2 className="text-xl font-semibold mb-2">{label} Error</h2>
    {process.env.NODE_ENV !== 'production' && error && (
      <pre className="text-left whitespace-pre-wrap text-sm text-red-600 mb-4">
        {error.message}
      </pre>
    )}
    <div className="space-x-3">
      <button
        onClick={reset}
        className="px-4 py-2 bg-blue-600 text-white rounded"
      >
        Retry
      </button>
      <a href="/" className="px-4 py-2 border rounded">Home</a>
    </div>
  </div>
);

export const DashboardErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ErrorBoundary
    level="dashboard"
    fallback={(reset, error) => DefaultFallback(reset, error, 'Dashboard')}
  >
    {children}
  </ErrorBoundary>
);

export const DocumentProcessingErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ErrorBoundary
    level="document-processing"
    fallback={(reset, error) => DefaultFallback(reset, error, 'Document Processing')}
  >
    {children}
  </ErrorBoundary>
);

export default ErrorBoundary;
