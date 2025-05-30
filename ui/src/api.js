/**
 * QuantumSpectre Elite Trading System
 * API Client
 * 
 * This module provides a unified API client for interacting with the backend services.
 * It handles authentication, request formatting, error handling, and WebSocket connections.
 */

import axios from 'axios';
import { io } from 'socket.io-client';
import { store } from './store';
import { authActions } from './slices/authSlice';
import { alertsActions } from './slices/alertsSlice';

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';
const WS_BASE_URL = process.env.REACT_APP_WS_BASE_URL || 'http://localhost:5000';

// Create axios instance with default config
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Socket.io instance
let socket = null;

/**
 * Socket connection manager
 */
export const socketManager = {
  connect() {
    if (socket) return socket;

    socket = io(WS_BASE_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
      autoConnect: true,
      query: {
        token: localStorage.getItem('token'),
      },
    });

    socket.on('connect', () => {
      console.log('WebSocket connected');
    });

    socket.on('disconnect', (reason) => {
      console.log(`WebSocket disconnected: ${reason}`);
    });

    socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      store.dispatch(alertsActions.addAlert({
        type: 'error',
        message: 'WebSocket connection error',
        details: error.message,
        timeout: 5000,
      }));
    });

    socket.on('reconnect', (attemptNumber) => {
      console.log(`WebSocket reconnected after ${attemptNumber} attempts`);
    });

    return socket;
  },
  
  disconnect() {
    if (socket) {
      socket.disconnect();
      socket = null;
    }
  },

  getSocket() {
    return socket || this.connect();
  },

  // Subscribe to specific events
  subscribe(event, callback) {
    const s = this.getSocket();
    s.on(event, callback);
    return () => s.off(event, callback);
  },

  // Emit events to the server
  emit(event, data) {
    const s = this.getSocket();
    return new Promise((resolve, reject) => {
      s.emit(event, data, (response) => {
        if (response && response.error) {
          reject(response.error);
        } else {
          resolve(response);
        }
      });
    });
  },
};

// Request interceptor for API calls
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for API calls
axiosInstance.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // Handle token refresh
    if (error.response && error.response.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Try to refresh token
        const refreshToken = localStorage.getItem('refreshToken');
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }
        
        const response = await axios.post(`${API_BASE_URL}/auth/refresh`, { 
          refreshToken 
        });
        
        if (response.data && response.data.token) {
          localStorage.setItem('token', response.data.token);
          localStorage.setItem('refreshToken', response.data.refreshToken);
          
          // Update auth state in Redux
          store.dispatch(authActions.refreshTokenSuccess({
            token: response.data.token,
            refreshToken: response.data.refreshToken,
          }));
          
          // Retry the original request with the new token
          originalRequest.headers['Authorization'] = `Bearer ${response.data.token}`;
          return axiosInstance(originalRequest);
        } else {
          throw new Error('Invalid refresh token response');
        }
      } catch (refreshError) {
        // If refresh fails, log out the user
        store.dispatch(authActions.logout());
        return Promise.reject(refreshError);
      }
    }
    
    // Add global error handling
    if (error.response) {
      // Server responded with an error status
      const errorMessage = error.response.data.message || 'Unknown server error';
      
      store.dispatch(alertsActions.addAlert({
        type: 'error',
        message: `API Error: ${errorMessage}`,
        details: `Status: ${error.response.status}`,
        timeout: 5000,
      }));
    } else if (error.request) {
      // Request was made but no response received
      store.dispatch(alertsActions.addAlert({
        type: 'error',
        message: 'Network Error',
        details: 'No response received from server',
        timeout: 5000,
      }));
    } else {
      // Something else happened
      store.dispatch(alertsActions.addAlert({
        type: 'error',
        message: 'Request Error',
        details: error.message,
        timeout: 5000,
      }));
    }
    
    return Promise.reject(error);
  }
);

/**
 * API client with methods for each endpoint
 */
export const api = {
  // Auth endpoints
  auth: {
    login: (credentials) => axiosInstance.post('/auth/login', credentials),
    register: (userData) => axiosInstance.post('/auth/register', userData),
    logout: () => axiosInstance.post('/auth/logout'),
    refreshToken: (refreshToken) => axiosInstance.post('/auth/refresh', { refreshToken }),
    getProfile: () => axiosInstance.get('/auth/profile'),
    updateProfile: (profileData) => axiosInstance.put('/auth/profile', profileData),
    changePassword: (passwordData) => axiosInstance.post('/auth/change-password', passwordData),
  },

  // Platform endpoints
  platform: {
    getStatus: () => axiosInstance.get('/platform/status'),
    getPlatforms: () => axiosInstance.get('/platform/list'),
    switchPlatform: (platformId) => axiosInstance.post('/platform/switch', { platformId }),
    getCredentials: (platformId) => axiosInstance.get(`/platform/${platformId}/credentials`),
    updateCredentials: (platformId, credentials) => 
      axiosInstance.post(`/platform/${platformId}/credentials`, credentials),
    testConnection: (platformId) => axiosInstance.post(`/platform/${platformId}/test-connection`),
  },

  // Market data endpoints
  market: {
    getSymbols: (platformId) => axiosInstance.get(`/market/${platformId}/symbols`),
    getCandles: (params) => axiosInstance.get('/market/candles', { params }),
    getOrderBook: (params) => axiosInstance.get('/market/orderbook', { params }),
    getTrades: (params) => axiosInstance.get('/market/trades', { params }),
    getTicker: (params) => axiosInstance.get('/market/ticker', { params }),
    getMarketSentiment: (symbol) => axiosInstance.get(`/market/sentiment/${symbol}`),
    getMarketNews: (params) => axiosInstance.get('/market/news', { params }),
  },

  // Trading endpoints
  trading: {
    placeOrder: (orderData) => axiosInstance.post('/trading/order', orderData),
    cancelOrder: (orderId) => axiosInstance.delete(`/trading/order/${orderId}`),
    modifyOrder: (orderId, orderData) => axiosInstance.put(`/trading/order/${orderId}`, orderData),
    getOpenOrders: (params) => axiosInstance.get('/trading/orders/open', { params }),
    getOrderHistory: (params) => axiosInstance.get('/trading/orders/history', { params }),
    getPositions: (params) => axiosInstance.get('/trading/positions', { params }),
    closePosition: (positionId) => axiosInstance.post(`/trading/position/${positionId}/close`),
    partialClose: (positionId, percentage) => 
      axiosInstance.post(`/trading/position/${positionId}/partial-close`, { percentage }),
  },

  // Brain and strategy endpoints
  brain: {
    getStrategies: () => axiosInstance.get('/brain/strategies'),
    getBrains: () => axiosInstance.get('/brain/list'),
    getBrainDetails: (brainId) => axiosInstance.get(`/brain/${brainId}`),
    updateBrainSettings: (brainId, settings) => axiosInstance.put(`/brain/${brainId}/settings`, settings),
    activateBrain: (brainId) => axiosInstance.post(`/brain/${brainId}/activate`),
    deactivateBrain: (brainId) => axiosInstance.post(`/brain/${brainId}/deactivate`),
    createBrain: (brainData) => axiosInstance.post('/brain/create', brainData),
    deleteBrain: (brainId) => axiosInstance.delete(`/brain/${brainId}`),
    getSignals: (params) => axiosInstance.get('/brain/signals', { params }),
  },

  // Auto trading endpoints
  autoTrading: {
    getStatus: () => axiosInstance.get('/autotrading/status'),
    enable: (settings) => axiosInstance.post('/autotrading/enable', settings),
    disable: () => axiosInstance.post('/autotrading/disable'),
    getSettings: () => axiosInstance.get('/autotrading/settings'),
    updateSettings: (settings) => axiosInstance.put('/autotrading/settings', settings),
  },

  // Portfolio and account endpoints
  portfolio: {
    getSummary: () => axiosInstance.get('/portfolio/summary'),
    getBalance: () => axiosInstance.get('/portfolio/balance'),
    getHistory: (params) => axiosInstance.get('/portfolio/history', { params }),
    getPerformance: (params) => axiosInstance.get('/portfolio/performance', { params }),
  },

  // System settings endpoints
  settings: {
    getAll: () => axiosInstance.get('/settings'),
    update: (settings) => axiosInstance.put('/settings', settings),
    getRiskSettings: () => axiosInstance.get('/settings/risk'),
    updateRiskSettings: (riskSettings) => axiosInstance.put('/settings/risk', riskSettings),
    getNotificationSettings: () => axiosInstance.get('/settings/notifications'),
    updateNotificationSettings: (notificationSettings) => 
      axiosInstance.put('/settings/notifications', notificationSettings),
    getUISettings: () => axiosInstance.get('/settings/ui'),
    updateUISettings: (uiSettings) => axiosInstance.put('/settings/ui', uiSettings),
  },

  // Backtester endpoints
  backtester: {
    run: (backtestConfig) => axiosInstance.post('/backtester/run', backtestConfig),
    getStatus: (backtestId) => axiosInstance.get(`/backtester/status/${backtestId}`),
    getResults: (backtestId) => axiosInstance.get(`/backtester/results/${backtestId}`),
    getSavedBacktests: () => axiosInstance.get('/backtester/saved'),
    saveBacktest: (backtestId, name) => 
      axiosInstance.post(`/backtester/${backtestId}/save`, { name }),
    deleteBacktest: (backtestId) => axiosInstance.delete(`/backtester/${backtestId}`),
  },

  // Voice advisor endpoints
  voiceAdvisor: {
    getStatus: () => axiosInstance.get('/voice-advisor/status'),
    enable: () => axiosInstance.post('/voice-advisor/enable'),
    disable: () => axiosInstance.post('/voice-advisor/disable'),
    getSettings: () => axiosInstance.get('/voice-advisor/settings'),
    updateSettings: (settings) => axiosInstance.put('/voice-advisor/settings', settings),
  },

  // System and monitoring endpoints
  system: {
    getStatus: () => axiosInstance.get('/system/status'),
    getLogs: (params) => axiosInstance.get('/system/logs', { params }),
    getMetrics: (params) => axiosInstance.get('/system/metrics', { params }),
  },
};

export default api;