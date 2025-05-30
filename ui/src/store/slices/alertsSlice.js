import { createSlice } from '@reduxjs/toolkit';

const slice = createSlice({
  name: 'alerts',
  initialState: {
    list: []
  },
  reducers: {
    addAlert: {
      reducer(state, action) {
        state.list.push(action.payload);
      },
      prepare(alert) {
        return {
          payload: {
            id: Date.now(),
            timestamp: Date.now(),
            severity: 'info',
            ...alert,
          },
        };
      }
    },
    removeAlert(state, action) {
      state.list = state.list.filter((a) => a.id !== action.payload);
    },
    clearAlerts(state) {
      state.list = [];
    }
  }
});

export const { actions } = slice;
export default slice;
