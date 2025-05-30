import React, { createContext, useContext } from 'react';

export const VoiceAdvisorContext = createContext({
  initializeVoiceAdvisor: () => {}
});

export const VoiceAdvisorProvider = ({ children }) => {
  const value = {
    initializeVoiceAdvisor: () => {}
  };

  return (
    <VoiceAdvisorContext.Provider value={value}>
      {children}
    </VoiceAdvisorContext.Provider>
  );
};

export const useVoiceAdvisorContext = () => useContext(VoiceAdvisorContext);
