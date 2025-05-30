import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { SnackbarProvider } from 'notistack';

// Contexts
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeModeProvider, useThemeMode } from './contexts/ThemeModeContext';
import { SystemMonitorProvider } from './contexts/SystemMonitorContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { WorkspaceProvider } from './contexts/WorkspaceContext';
import { VoiceAdvisorProvider } from './contexts/VoiceAdvisorContext';

// Components
import MainLayout from './layouts/MainLayout';
import LoadingScreen from './components/common/LoadingScreen';
import ErrorBoundary from './components/common/ErrorBoundary';

// Pages
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import ResetPassword from './pages/auth/ResetPassword';
import VerifyEmail from './pages/auth/VerifyEmail';
import Dashboard from './pages/Dashboard';
import TradingTerminal from './pages/TradingTerminal';
import Portfolio from './pages/Portfolio';
import MarketAnalysis from './pages/MarketAnalysis';
import BrainPerformance from './pages/BrainPerformance';
import PatternLibrary from './pages/PatternLibrary';
import MlModelTraining from './pages/MlModelTraining';
import Backtesting from './pages/Backtesting';
import StrategyBuilder from './pages/StrategyBuilder';
import SystemMonitor from './pages/SystemMonitor';
import Notifications from './pages/Notifications';
import Settings from './pages/Settings';
import EnhancedIntelligence from './pages/EnhancedIntelligence';

// Protected route component
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <LoadingScreen />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  return children;
};

const AppContent = () => {
  const { theme } = useThemeMode();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <ErrorBoundary>
          <Routes>
            {/* Auth routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/reset-password" element={<ResetPassword />} />
            <Route path="/verify-email" element={<VerifyEmail />} />

            {/* Protected routes */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <MainLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<Dashboard />} />
              <Route path="trading" element={<TradingTerminal />} />
              <Route path="portfolio" element={<Portfolio />} />
              <Route path="enhanced-intelligence" element={<EnhancedIntelligence />} />
              <Route path="market-analysis" element={<MarketAnalysis />} />
              <Route path="brain-performance" element={<BrainPerformance />} />
              <Route path="pattern-library" element={<PatternLibrary />} />
              <Route path="ml-model-training" element={<MlModelTraining />} />
              <Route path="backtesting" element={<Backtesting />} />
              <Route path="strategy-builder" element={<StrategyBuilder />} />
              <Route path="system-monitor" element={<SystemMonitor />} />
              <Route path="notifications" element={<Notifications />} />
              <Route path="settings" element={<Settings />} />
            </Route>

            {/* Fallback route */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ErrorBoundary>
      </Router>
    </ThemeProvider>
  );
};

const App = () => {
  return (
    <ErrorBoundary>
      <SnackbarProvider maxSnack={3}>
        <ThemeModeProvider>
          <AuthProvider>
            <SystemMonitorProvider>
              <WebSocketProvider>
                <WorkspaceProvider>
                  <VoiceAdvisorProvider>
                    <AppContent />
                  </VoiceAdvisorProvider>
                </WorkspaceProvider>
              </WebSocketProvider>
            </SystemMonitorProvider>
          </AuthProvider>
        </ThemeModeProvider>
      </SnackbarProvider>
    </ErrorBoundary>
  );
};

export default App;
