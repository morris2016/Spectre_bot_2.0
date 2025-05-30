import React, { createContext, useContext } from 'react';

export const WorkspaceContext = createContext({
  saveWorkspace: () => {},
  loadWorkspace: () => {}
});

export const WorkspaceProvider = ({ children }) => {
  const value = {
    saveWorkspace: () => {},
    loadWorkspace: () => {}
  };
  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
};

export const useWorkspaceContext = () => useContext(WorkspaceContext);
