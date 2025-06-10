// @ts-nocheck
import React, { useState, useEffect, createContext, useContext } from 'react';
import {
  Search, Upload, FileText, Users, Settings, Shield,
  Activity, Database, Cpu, AlertCircle, CheckCircle,
  Play, Pause, RefreshCw, ChevronRight, Home,
  BarChart, Network, Workflow, Eye, Download,
  Clock, Filter, Plus, Trash2, Edit, Save
} from 'lucide-react';
import ErrorBoundary from '../../frontend/src/components/ErrorBoundary';
import {
  DashboardErrorBoundary,
  DocumentProcessingErrorBoundary,
} from '../../frontend/src/components/AsyncErrorBoundary';
import { Button } from '../../frontend/src/design-system';



// Context for global state management
const AppContext = createContext({});

// Main App Component
export default function LegalAISystem() {
  const [currentView, setCurrentView] = useState('dashboard');
  const [notifications, setNotifications] = useState([]);
  const [systemStatus, setSystemStatus] = useState({
    services: { total: 12, healthy: 10, unhealthy: 2 },
    documents: { processing: 3, completed: 156, failed: 2 },
    performance: { cpu: 45, memory: 62, disk: 38 }
  });

  const addNotification = (message, type = 'info') => {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, message, type, timestamp: new Date() }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  const contextValue = {
    systemStatus,
    setSystemStatus,
    addNotification,
    currentView,
    setCurrentView
  };

  return (
    <ErrorBoundary level="critical">
      <AppContext.Provider value={contextValue}>
        <div className="min-h-screen bg-gray-50">
          <Sidebar />
          <div className="ml-64">
            <Header />
            <main className="p-6">
              <NotificationArea notifications={notifications} />
              {currentView === 'dashboard' && (
                <DashboardErrorBoundary>
                  <Dashboard />
                </DashboardErrorBoundary>
              )}
              {currentView === 'documents' && (
                <DocumentProcessingErrorBoundary>
                  <DocumentProcessing />
                </DocumentProcessingErrorBoundary>
              )}
              {currentView === 'knowledge' && (
                <ErrorBoundary level="view">
                  <KnowledgeGraph />
                </ErrorBoundary>
              )}
              {currentView === 'agents' && (
                <ErrorBoundary level="view">
                  <AgentManagement />
                </ErrorBoundary>
              )}
              {currentView === 'workflows' && (
                <ErrorBoundary level="view">
                  <WorkflowDesigner />
                </ErrorBoundary>
              )}
              {currentView === 'monitoring' && (
                <ErrorBoundary level="view">
                  <Monitoring />
                </ErrorBoundary>
              )}
              {currentView === 'security' && (
                <ErrorBoundary level="view">
                  <SecurityManagement />
                </ErrorBoundary>
              )}
              {currentView === 'settings' && (
                <ErrorBoundary level="view">
                  <SystemSettings />
                </ErrorBoundary>
              )}
            </main>
          </div>
        </div>
      </AppContext.Provider>
    </ErrorBoundary>
  );
}

// Sidebar Navigation
function Sidebar() {
  const { currentView, setCurrentView } = useContext(AppContext);
  
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'documents', label: 'Document Processing', icon: FileText },
    { id: 'knowledge', label: 'Knowledge Graph', icon: Network },
    { id: 'agents', label: 'Agent Management', icon: Cpu },
    { id: 'workflows', label: 'Workflow Designer', icon: Workflow },
    { id: 'monitoring', label: 'Monitoring', icon: Activity },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  return (
    <div className="fixed left-0 top-0 h-full w-64 bg-gray-900 text-white">
      <div className="p-4">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Shield className="w-8 h-8" />
          Legal AI System
        </h1>
      </div>
      
      <nav className="mt-8">
        {menuItems.map(item => (
          <Button
            key={item.id}
            onClick={() => setCurrentView(item.id)}
            variant="secondary"
            className={`w-full px-4 py-3 flex items-center gap-3 hover:bg-gray-800 transition-colors ${
              currentView === item.id ? 'bg-gray-800 border-l-4 border-blue-500' : ''
            }`}
          >
            <item.icon className="w-5 h-5" />
            <span>{item.label}</span>
          </Button>
        ))}
      </nav>
    </div>
  );
}

// Header Component
function Header() {
  const { systemStatus } = useContext(AppContext);
  const healthPercentage = (systemStatus.services.healthy / systemStatus.services.total) * 100;

  return (
    <header className="bg-white shadow-sm border-b">
      <div className="px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="text-sm">
            <span className="text-gray-500">System Health:</span>
            <span className={`ml-2 font-semibold ${healthPercentage > 80 ? 'text-green-600' : 'text-yellow-600'}`}>
              {healthPercentage.toFixed(0)}%
            </span>
          </div>
          <div className="text-sm">
            <span className="text-gray-500">Active Documents:</span>
            <span className="ml-2 font-semibold">{systemStatus.documents.processing}</span>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <Button variant="outline" size="sm" className="p-2 hover:bg-gray-100 rounded-lg">
            <RefreshCw className="w-5 h-5" />
          </Button>
          <Button variant="outline" size="sm" className="p-2 hover:bg-gray-100 rounded-lg relative">
            <AlertCircle className="w-5 h-5" />
            <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
              2
            </span>
          </Button>
          <div className="flex items-center gap-2">
            <img src="/api/placeholder/32/32" alt="User" className="w-8 h-8 rounded-full" />
            <span className="text-sm font-medium">Admin User</span>
          </div>
        </div>
      </div>
    </header>
  );
}

// Notification Area
function NotificationArea({ notifications }) {
  return (
    <div className="fixed top-20 right-6 z-50 space-y-2">
      {notifications.map(notif => (
        <div
          key={notif.id}
          className={`bg-white shadow-lg rounded-lg p-4 flex items-center gap-3 transform transition-all ${
            notif.type === 'success' ? 'border-l-4 border-green-500' :
            notif.type === 'error' ? 'border-l-4 border-red-500' :
            'border-l-4 border-blue-500'
          }`}
        >
          {notif.type === 'success' ? <CheckCircle className="w-5 h-5 text-green-500" /> :
           notif.type === 'error' ? <AlertCircle className="w-5 h-5 text-red-500" /> :
           <AlertCircle className="w-5 h-5 text-blue-500" />}
          <span className="text-sm">{notif.message}</span>
        </div>
      ))}
    </div>
  );
}

// Dashboard Component
function Dashboard() {
  const { systemStatus, setSystemStatus } = useContext(AppContext);
  const idRef = React.useRef<string>('client-' + Math.random().toString(36).slice(2));
  const { status, connected } = useRealtimeSystemStatus(idRef.current);

  React.useEffect(() => {
    if (status) {
      setSystemStatus(prev => ({
        ...prev,
        performance: { cpu: status.cpu, memory: status.memory, disk: status.disk }
      }));
    }
  }, [status]);
  
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        System Dashboard
        <span className={`text-sm ${connected ? 'text-green-600' : 'text-red-600'}`}>{connected ? 'connected' : 'disconnected'}</span>
      </h2>
      
      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatusCard
          title="Services Status"
          value={`${systemStatus.services.healthy}/${systemStatus.services.total}`}
          subtitle="Services Running"
          icon={<Cpu className="w-8 h-8 text-blue-500" />}
          trend="+2.5%"
        />
        <StatusCard
          title="Document Processing"
          value={systemStatus.documents.completed}
          subtitle="Documents Processed"
          icon={<FileText className="w-8 h-8 text-green-500" />}
          trend="+12%"
        />
        <StatusCard
          title="System Performance"
          value={`${systemStatus.performance.cpu}%`}
          subtitle="CPU Usage"
          icon={<Activity className="w-8 h-8 text-purple-500" />}
          trend="-5%"
        />
      </div>

      {/* Performance Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">System Performance</h3>
        <div className="h-64 bg-gray-100 rounded flex items-center justify-center">
          <BarChart className="w-12 h-12 text-gray-400" />
          <span className="ml-2 text-gray-500">Performance Chart</span>
        </div>
      </div>

      {/* Recent Activities */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Recent Activities</h3>
        <div className="space-y-3">
          {[
            { action: 'Document Processed', file: 'case_2024_001.pdf', time: '5 min ago', status: 'success' },
            { action: 'Entity Extraction', file: 'deposition_transcript.docx', time: '12 min ago', status: 'success' },
            { action: 'Workflow Failed', file: 'evidence_log.xlsx', time: '1 hour ago', status: 'error' }
          ].map((activity, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 hover:bg-gray-50 rounded">
              <div className="flex items-center gap-3">
                <div className={`w-2 h-2 rounded-full ${activity.status === 'success' ? 'bg-green-500' : 'bg-red-500'}`} />
                <div>
                  <div className="font-medium">{activity.action}</div>
                  <div className="text-sm text-gray-500">{activity.file}</div>
                </div>
              </div>
              <span className="text-sm text-gray-500">{activity.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Document Processing Interface
function DocumentProcessing() {
  const { addNotification } = useContext(AppContext);
  const { isLoading, error, data: documents, executeAsync } =
    useLoadingState<Array<{ id: number; name: string; status: string; progress: number }>>();

  useEffect(() => {
    executeAsync(() =>
      fetch('data/sample_documents.json').then(res => {
        if (!res.ok) {
          throw new Error('Failed to load documents');
        }
        return res.json();
      })
    );
  }, []);

  const handleFileUpload = () => {
    addNotification('File uploaded successfully', 'success');
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Document Processing</h2>
        <div className="flex gap-3">
          <Button className="px-4 py-2 flex items-center gap-2" variant="primary">
            <Upload className="w-4 h-4" />
            Upload Documents
          </Button>
          <Button className="px-4 py-2 border border-gray-300 flex items-center gap-2" variant="outline">
            <Filter className="w-4 h-4" />
            Filter
          </Button>
        </div>
      </div>

      {/* Upload Area */}
      <div className="bg-white rounded-lg shadow p-8 border-2 border-dashed border-gray-300 text-center">
        <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600 mb-2">Drag and drop files here or click to browse</p>
        <p className="text-sm text-gray-500">Supported formats: PDF, DOCX, TXT, CSV, XLSX</p>
        <Button
          onClick={handleFileUpload}
          className="mt-4 px-6 py-2"
          variant="primary"
        >
          Select Files
        </Button>
      </div>

      {/* Document Queue */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold">Processing Queue</h3>
        </div>
        <div className="p-6 space-y-4">
          {isLoading && <TableSkeleton rows={3} columns={3} />}
          {error && (
            <div className="text-red-600 space-x-2">
              <span>Error loading documents</span>
              <Button
                variant="outline"
                size="sm"
                style={{ textDecoration: 'underline' }}
                onClick={() =>
                  executeAsync(() =>
                    fetch('data/sample_documents.json').then(res => {
                      if (!res.ok) throw new Error('Failed to load documents');
                      return res.json();
                    })
                  )
                }
              >
                Retry
              </Button>
            </div>
          )}
          {documents &&
            documents.map(doc => (
            <div key={doc.id} className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-gray-500" />
                  <span className="font-medium">{doc.name}</span>
                  <span className={`text-sm px-2 py-1 rounded ${
                    doc.status === 'completed' ? 'bg-green-100 text-green-700' :
                    doc.status === 'processing' ? 'bg-blue-100 text-blue-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {doc.status}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" className="p-1 hover:bg-gray-100 rounded">
                    <Eye className="w-4 h-4" />
                  </Button>
                  <Button variant="outline" size="sm" className="p-1 hover:bg-gray-100 rounded">
                    <Download className="w-4 h-4" />
                  </Button>
                  <Button variant="outline" size="sm" className="p-1 hover:bg-gray-100 rounded text-red-500">
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              {doc.status === 'processing' && (
                <div className="mt-2">
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span>Processing</span>
                    <span>{doc.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all"
                      style={{ width: `${doc.progress}%` }}
                    />
                  </div>
                  <div className="mt-2">
                    <ProgressiveLoader
                      stages={[
                        { label: 'Uploaded', completed: true },
                        { label: 'Processing', completed: doc.progress === 100 }
                      ]}
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Knowledge Graph Viewer
function KnowledgeGraph() {
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [filterType, setFilterType] = useState('all');

  const entities = [
    { id: 1, name: 'John Doe', type: 'Person', connections: 5 },
    { id: 2, name: 'ABC Corporation', type: 'Organization', connections: 8 },
    { id: 3, name: 'Case 2024-001', type: 'Case', connections: 12 }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Knowledge Graph</h2>
        <div className="flex gap-3">
          <select 
            className="px-4 py-2 border border-gray-300 rounded-lg"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
          >
            <option value="all">All Entities</option>
            <option value="person">Persons</option>
            <option value="organization">Organizations</option>
            <option value="case">Cases</option>
          </select>
          <Button className="px-4 py-2" variant="primary">
            Export Graph
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Graph Visualization */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
          <div className="h-96 bg-gray-100 rounded flex items-center justify-center">
            <Network className="w-16 h-16 text-gray-400" />
            <span className="ml-4 text-gray-500">Interactive Graph Visualization</span>
          </div>
        </div>

        {/* Entity Details */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Entity Explorer</h3>
          <div className="space-y-3">
            {entities.map(entity => (
              <div 
                key={entity.id}
                onClick={() => setSelectedEntity(entity)}
                className={`p-3 border rounded-lg cursor-pointer hover:bg-gray-50 ${
                  selectedEntity?.id === entity.id ? 'border-blue-500 bg-blue-50' : ''
                }`}
              >
                <div className="font-medium">{entity.name}</div>
                <div className="text-sm text-gray-500">
                  {entity.type} • {entity.connections} connections
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Entity Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard title="Total Entities" value="1,234" icon={<Users />} />
        <StatCard title="Relationships" value="3,456" icon={<Network />} />
        <StatCard title="Documents" value="567" icon={<FileText />} />
        <StatCard title="Violations Found" value="23" icon={<AlertCircle />} />
      </div>
    </div>
  );
}

// Agent Management Interface
function AgentManagement() {
  const [agents] = useState([
    { id: 1, name: 'Document Processor', status: 'running', cpu: 23, memory: 45, tasks: 12 },
    { id: 2, name: 'Entity Extractor', status: 'running', cpu: 67, memory: 78, tasks: 8 },
    { id: 3, name: 'Legal Analyzer', status: 'stopped', cpu: 0, memory: 0, tasks: 0 },
    { id: 4, name: 'Violation Detector', status: 'running', cpu: 45, memory: 56, tasks: 5 }
  ]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Agent Management</h2>
        <Button className="px-4 py-2" variant="primary">
          Deploy New Agent
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {agents.map(agent => (
          <div key={agent.id} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Cpu className="w-8 h-8 text-blue-500" />
                <div>
                  <h3 className="font-semibold">{agent.name}</h3>
                  <span className={`text-sm ${
                    agent.status === 'running' ? 'text-green-600' : 'text-gray-500'
                  }`}>
                    {agent.status}
                  </span>
                </div>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className={`p-2 rounded ${agent.status === 'running' ? 'hover:bg-gray-100' : 'bg-green-100 hover:bg-green-200'}`}
                >
                  {agent.status === 'running' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </Button>
                <Button variant="outline" size="sm" className="p-2 hover:bg-gray-100 rounded">
                  <Settings className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>CPU Usage</span>
                  <span>{agent.cpu}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${agent.cpu}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Memory Usage</span>
                  <span>{agent.memory}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full"
                    style={{ width: `${agent.memory}%` }}
                  />
                </div>
              </div>
              <div className="pt-2 border-t">
                <div className="text-sm text-gray-600">
                  Active Tasks: <span className="font-semibold">{agent.tasks}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


// Monitoring Dashboard
function Monitoring() {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">System Monitoring</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Real-time Metrics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Real-time Metrics</h3>
          <div className="space-y-4">
            <MetricRow label="API Response Time" value="124ms" status="good" />
            <MetricRow label="Database Queries/sec" value="847" status="good" />
            <MetricRow label="Error Rate" value="0.02%" status="warning" />
            <MetricRow label="Active Connections" value="234" status="good" />
          </div>
        </div>

        {/* System Logs */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">System Logs</h3>
          <div className="space-y-2 font-mono text-sm">
            {[
              { level: 'INFO', message: 'Document processed successfully', time: '12:34:56' },
              { level: 'WARN', message: 'High memory usage detected', time: '12:33:45' },
              { level: 'ERROR', message: 'Failed to connect to vector store', time: '12:32:12' },
              { level: 'INFO', message: 'Entity extraction completed', time: '12:31:00' }
            ].map((log, idx) => (
              <div key={idx} className="flex items-center gap-2 text-xs">
                <span className="text-gray-500">{log.time}</span>
                <span className={`px-2 py-1 rounded ${
                  log.level === 'ERROR' ? 'bg-red-100 text-red-700' :
                  log.level === 'WARN' ? 'bg-yellow-100 text-yellow-700' :
                  'bg-blue-100 text-blue-700'
                }`}>
                  {log.level}
                </span>
                <span>{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Performance Charts */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Performance Trends</h3>
        <div className="h-64 bg-gray-100 rounded flex items-center justify-center">
          <Activity className="w-12 h-12 text-gray-400" />
          <span className="ml-2 text-gray-500">Performance Charts</span>
        </div>
      </div>
    </div>
  );
}

// Security Management
function SecurityManagement() {
  const [users] = useState([
    { id: 1, name: 'Admin User', email: 'admin@legal-ai.com', role: 'Administrator', status: 'active' },
    { id: 2, name: 'Legal Analyst', email: 'analyst@legal-ai.com', role: 'Analyst', status: 'active' },
    { id: 3, name: 'Guest User', email: 'guest@legal-ai.com', role: 'Viewer', status: 'inactive' }
  ]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Security Management</h2>
        <Button className="px-4 py-2" variant="primary">
          Add User
        </Button>
      </div>

      {/* Security Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <Shield className="w-8 h-8 text-green-500" />
            <span className="text-2xl font-bold">98.5%</span>
          </div>
          <h3 className="font-semibold">Security Score</h3>
          <p className="text-sm text-gray-500">Last audit: 2 days ago</p>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <Users className="w-8 h-8 text-blue-500" />
            <span className="text-2xl font-bold">12</span>
          </div>
          <h3 className="font-semibold">Active Users</h3>
          <p className="text-sm text-gray-500">3 administrators</p>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <AlertCircle className="w-8 h-8 text-yellow-500" />
            <span className="text-2xl font-bold">2</span>
          </div>
          <h3 className="font-semibold">Security Alerts</h3>
          <p className="text-sm text-gray-500">Action required</p>
        </div>
      </div>

      {/* User Management */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold">User Management</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Role</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {users.map(user => (
                <tr key={user.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div>
                      <div className="text-sm font-medium text-gray-900">{user.name}</div>
                      <div className="text-sm text-gray-500">{user.email}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                      {user.role}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      user.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                    }`}>
                      {user.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <Button className="text-indigo-600 hover:text-indigo-900 mr-3" variant="outline" size="sm">Edit</Button>
                    <Button className="text-red-600 hover:text-red-900" variant="outline" size="sm">Delete</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// System Settings
function SystemSettings() {
  const { addNotification } = useContext(AppContext);
  
  const handleSave = () => {
    addNotification('Settings saved successfully', 'success');
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">System Settings</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* General Settings */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">General Settings</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">System Name</label>
              <input type="text" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="Legal AI System" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Default Language</label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg">
                <option>English</option>
                <option>Spanish</option>
                <option>French</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Time Zone</label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg">
                <option>UTC</option>
                <option>EST</option>
                <option>PST</option>
              </select>
            </div>
          </div>
        </div>

        {/* Processing Settings */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Processing Settings</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Concurrent Documents</label>
              <input type="number" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="5" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Chunk Size</label>
              <input type="number" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="3000" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confidence Threshold</label>
              <input type="number" step="0.1" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="0.7" />
            </div>
          </div>
        </div>

        {/* LLM Settings */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">LLM Provider Settings</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Primary Provider</label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg">
                <option>xAI (Grok)</option>
                <option>OpenAI</option>
                <option>Ollama (Local)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg">
                <option>grok-3-mini</option>
                <option>grok-3-reasoning</option>
              </select>
            </div>
            <div>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" defaultChecked />
                <span className="text-sm">Enable fallback providers</span>
              </label>
            </div>
          </div>
        </div>

        {/* Storage Settings */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Storage Settings</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Vector Store Path</label>
              <input type="text" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="./storage/vectors" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Database URL</label>
              <input type="text" className="w-full px-3 py-2 border border-gray-300 rounded-lg" defaultValue="postgresql://localhost:5432/legal_ai" />
            </div>
            <div>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" defaultChecked />
                <span className="text-sm">Enable caching</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-end gap-3">
        <Button className="px-4 py-2 border border-gray-300" variant="outline">
          Cancel
        </Button>
        <Button onClick={handleSave} className="px-4 py-2" variant="primary">
          Save Settings
        </Button>
      </div>
    </div>
  );
}

// Helper Components
function StatusCard({ title, value, subtitle, icon, trend }) {
  const isPositive = trend?.startsWith('+');
  
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">{value}</p>
          <p className="text-sm text-gray-500">{subtitle}</p>
          {trend && (
            <p className={`text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {trend} from last week
            </p>
          )}
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          {icon}
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, icon }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600">{title}</p>
          <p className="text-xl font-semibold">{value}</p>
        </div>
        <div className="text-gray-400">
          {icon}
        </div>
      </div>
    </div>
  );
}

function MetricRow({ label, value, status }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-600">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-semibold">{value}</span>
        <div className={`w-2 h-2 rounded-full ${
          status === 'good' ? 'bg-green-500' :
          status === 'warning' ? 'bg-yellow-500' :
          'bg-red-500'
        }`} />
      </div>
    </div>
  );
}