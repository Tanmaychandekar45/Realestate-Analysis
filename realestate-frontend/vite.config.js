import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // CRITICAL FIX: Explicitly define environment variables 
  // to prevent libraries from falling back to dynamic code generation (like eval)
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
  },
  server: {
    // Standard frontend port
    port: 3000, 
    host: true,
  },
})