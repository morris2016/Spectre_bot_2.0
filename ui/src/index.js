/**
 * QuantumSpectre Elite Trading System
 * UI Entry Point
 *
 * This file serves as the entry point for the React application,
 * rendering the main App component to the DOM.
 */

import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { BrowserRouter } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';

import App from './App';
import store from './store';
import { theme } from './theme';
import { WorkspaceProvider } from './contexts/WorkspaceContext';
import { AuthProvider } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { VoiceAdvisorProvider } from './contexts/VoiceAdvisorContext';
import './i18n';
import './index.css';

// Initialize performance monitoring
import './utils/performance';

// Setup global error handling
import './utils/errorHandler';

// Register Service Worker for PWA support
import { registerSW } from './serviceWorker';

// Prepare GPU acceleration if available
import { initializeGPUAcceleration } from './utils/gpuAcceleration';

initializeGPUAcceleration();

ReactDOM.render(
  <React.StrictMode>
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          <WorkspaceProvider>
            <AuthProvider>
              <WebSocketProvider>
                <VoiceAdvisorProvider>
                  <SnackbarProvider maxSnack={3}>
                    <App />
                  </SnackbarProvider>
                </VoiceAdvisorProvider>
              </WebSocketProvider>
            </AuthProvider>
          </WorkspaceProvider>
        </BrowserRouter>
      </ThemeProvider>
    </Provider>
  </React.StrictMode>,
  document.getElementById('root')
);

// Register service worker for offline capabilities and PWA
registerSW();

// Add performance mark for initial render
performance.mark('app-rendered');
