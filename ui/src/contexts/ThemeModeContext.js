import React, { createContext, useContext, useState } from 'react';

export const ThemeModeContext = createContext({ mode: 'light' });

export const ThemeModeProvider = ({ children }) => {
  const [mode, setMode] = useState('light');
  const toggleMode = () => setMode(m => (m === 'light' ? 'dark' : 'light'));

  return (
    <ThemeModeContext.Provider value={{ mode, toggleMode }}>
      {children}
    </ThemeModeContext.Provider>
  );
};

export const useThemeModeContext = () => useContext(ThemeModeContext);
