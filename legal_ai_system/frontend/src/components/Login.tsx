import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import Button from '../design-system/components/Button';
import Input from '../design-system/components/Input';

const Login: React.FC = () => {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      await login(username, password);
    } catch {
      setError('Invalid credentials');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center">
      <form onSubmit={handleSubmit} className="bg-white p-6 shadow rounded space-y-4">
        <h2 className="text-xl font-bold">Login</h2>
        {error && <div className="text-red-600">{error}</div>}
        <Input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" />
        <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
        <Button type="submit" variant="primary" className="w-full">Sign In</Button>
      </form>
    </div>
  );
};

export default Login;
