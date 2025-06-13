import React from 'react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  /**
   * Optional fallback content to render when an error occurs. If a function is
   * provided it will receive a reset callback and the caught error.
   */
  fallback?: React.ReactNode | ((reset: () => void, error: Error | null) => React.ReactNode);
  /**
   * Optional category or severity for logging purposes.
   */
  level?: string;
  /**
   * Optional callback that fires when an error is caught. This allows
   * integration with external logging services.
   */
  onError?: (error: Error, info: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, error: null };

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    this.setState({ hasError: true, error });
    // Basic logging; replace with real logging service as needed
    if (this.props.onError) {
      this.props.onError(error, info);
    } else if (process.env.NODE_ENV !== 'production') {
      // eslint-disable-next-line no-console
      console.error(`ErrorBoundary [${this.props.level ?? 'unknown'}]`, error, info);
    }
  }

  reset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (typeof this.props.fallback === 'function') {
        return (this.props.fallback as any)(this.reset, this.state.error);
      }
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="p-8 text-center">
          <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
          {process.env.NODE_ENV !== 'production' && (
            <pre className="text-left whitespace-pre-wrap text-sm text-red-600 mb-4">
              {this.state.error?.message}
            </pre>
          )}
          <div className="space-x-3">
            <button onClick={this.reset} className="px-4 py-2 bg-blue-600 text-white rounded">Retry</button>
            <a href="/" className="px-4 py-2 border rounded">Home</a>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
