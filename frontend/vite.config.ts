import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'legal-ai-gui': path.resolve(__dirname, '../legal_ai_system/frontend/legal-ai-gui.tsx'),
      'lucide-react': path.resolve(__dirname, 'node_modules/lucide-react'),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
