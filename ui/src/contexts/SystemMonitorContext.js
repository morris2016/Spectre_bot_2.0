import React, { createContext, useContext } from 'react';

export const SystemMonitorContext = createContext({
  startMonitoring: () => {}
});

export const SystemMonitorProvider = ({ children }) => {
  const value = {
    startMonitoring: () => {}
  };

  return (
    <SystemMonitorContext.Provider value={value}>
      {children}
    </SystemMonitorContext.Provider>
  );
};

export const useSystemMonitorContext = () => useContext(SystemMonitorContext);
