import React, { createContext, useContext, useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { socketManager } from '../api';

export const WebSocketContext = createContext({
  socket: null,
  connect: () => {},
  disconnect: () => {},
});

export const WebSocketProvider = ({ children }) => {
  const token = useSelector((state) => state.auth.token);
  const [socket, setSocket] = useState(null);

  const connect = () => {
    const s = socketManager.connect();
    setSocket(s);
    return s;
  };

  const disconnect = () => {
    socketManager.disconnect();
    setSocket(null);
  };

  useEffect(() => {
    if (token) {
      connect();
      return () => disconnect();
    }
    return undefined;
  }, [token]);

  const value = { socket, connect, disconnect };

  return (
    <WebSocketContext.Provider value={value}>{children}</WebSocketContext.Provider>
  );
};

export const useWebSocketContext = () => useContext(WebSocketContext);
