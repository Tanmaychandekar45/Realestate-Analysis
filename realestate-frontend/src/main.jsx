import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './App.css'; // CRITICAL: Imports Tailwind CSS directives

// Use ReactDOM.createRoot for modern React 18 rendering
// This mounts the App component into the <div id="root"> in index.html
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);