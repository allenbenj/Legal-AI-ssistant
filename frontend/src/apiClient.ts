export default function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const token = localStorage.getItem('token');
  if (!token) {
    return Promise.reject(new Error('No auth token'));
  }
  const headers = new Headers(init.headers || {});
  headers.set('Authorization', `Bearer ${token}`);
  return fetch(input, { ...init, headers });
}
