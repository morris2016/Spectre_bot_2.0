/**
 * QuantumSpectre Elite Trading System
 * Redux Store Configuration
 * 
 * This module configures the Redux store for the application, including
 * all reducers, middleware, and state persistence.
 */

import { configureStore, combineReducers } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import thunk from 'redux-thunk';
import { createLogger } from 'redux-logger';
import { createEpicMiddleware } from 'redux-observable';
import { rootEpic } from './epics';

// Import slices
import authSlice from './slices/authSlice';
import marketDataSlice from './slices/marketDataSlice';
import tradingSlice from './slices/tradingSlice';
import alertsSlice from './slices/alertsSlice';
import portfolioSlice from './slices/portfolioSlice';
import settingsSlice from './slices/settingsSlice';
import uiStateSlice from './slices/uiStateSlice';
import platformSlice from './slices/platformSlice';
import assetSlice from './slices/assetSlice';
import brainSlice from './slices/brainSlice';
import signalSlice from './slices/signalSlice';
import dashboardSlice from './slices/dashboardSlice';
import strategySlice from './slices/strategySlice';
import backtesterSlice from './slices/backtesterSlice';
import voiceAdvisorSlice from './slices/voiceAdvisorSlice';
import systemSlice from './slices/systemSlice';
import preferencesSlice from './slices/preferencesSlice';

// Configure persistence
const persistConfig = {
  key: 'quantumspectre-root',
  storage,
  whitelist: ['auth', 'settings', 'portfolio'], // Only persist these reducers
  blacklist: ['marketData'], // Never persist these reducers
  transforms: [], // Add any transforms here
};

// Define reducers
const rootReducer = combineReducers({
  auth: authSlice.reducer,
  marketData: marketDataSlice.reducer,
  trading: tradingSlice.reducer,
  alerts: alertsSlice.reducer,
  portfolio: portfolioSlice.reducer,
  settings: settingsSlice.reducer,
  uiState: uiStateSlice.reducer,
  platform: platformSlice.reducer,
  asset: assetSlice.reducer,
  brain: brainSlice.reducer,
  signal: signalSlice.reducer,
  dashboard: dashboardSlice.reducer,
  strategy: strategySlice.reducer,
  backtester: backtesterSlice.reducer,
  voiceAdvisor: voiceAdvisorSlice.reducer,
  system: systemSlice.reducer,
  preferences: preferencesSlice.reducer,
});

// Create the persisted reducer
const persistedReducer = persistReducer(persistConfig, rootReducer);

// Configure middleware
const epicMiddleware = createEpicMiddleware();

// Create logger middleware with custom options
const loggerMiddleware = createLogger({
  collapsed: true,
  diff: true,
  duration: true,
  colors: {
    title: () => '#3B82F6', // blue
    prevState: () => '#9CA3AF', // gray
    action: () => '#EC4899', // pink
    nextState: () => '#10B981', // green
    error: () => '#EF4444', // red
  },
  predicate: (getState, action) => 
    !action.type.includes('marketData/updateTick') && // Filter out high-frequency updates
    !action.type.includes('@@redux'),
});

// Define middleware based on environment
const getMiddleware = () => {
  if (process.env.NODE_ENV === 'development') {
    return [thunk, epicMiddleware, loggerMiddleware];
  }
  return [thunk, epicMiddleware];
};

// Configure store with performance optimizations
const store = configureStore({
  reducer: persistedReducer,
  middleware: getMiddleware(),
  devTools: process.env.NODE_ENV !== 'production',
  enhancers: [],
  preloadedState: {},
});

// Run the root epic
epicMiddleware.run(rootEpic);

// Create persistor
export const persistor = persistStore(store);

// Export typed hooks for selectors and dispatching actions
export const { dispatch, getState } = store;

// Export store and action creators
export { store };
export {
  actions as authActions,
  login as loginUser,
  logout as logoutUser,
  checkAuthStatus,
} from './slices/authSlice';
export { actions as marketDataActions } from './slices/marketDataSlice';
export { actions as tradingActions } from './slices/tradingSlice';
export { actions as alertsActions } from './slices/alertsSlice';
export { actions as portfolioActions } from './slices/portfolioSlice';
export { actions as settingsActions } from './slices/settingsSlice';
export { actions as uiStateActions } from './slices/uiStateSlice';
export { actions as platformActions } from './slices/platformSlice';
export { actions as assetActions, fetchAssets } from './slices/assetSlice';
export { actions as brainActions } from './slices/brainSlice';
export { actions as signalActions } from './slices/signalSlice';
export { actions as dashboardActions, loadDashboard } from './slices/dashboardSlice';
export { actions as strategyActions } from './slices/strategySlice';
export { actions as backtesterActions } from './slices/backtesterSlice';
export { actions as voiceAdvisorActions } from './slices/voiceAdvisorSlice';
export {
  actions as systemActions,
  initializeSystem,
  checkSystemHealth,
} from './slices/systemSlice';
export { actions as preferencesActions, initializePreferences } from './slices/preferencesSlice';

// Create typed hooks
export const useAppSelector = (selector) => selector(getState());

// Initialize the store with required actions
store.dispatch(settingsActions.initializeSettings());
store.dispatch(uiStateActions.initializeUI());

export default store;
