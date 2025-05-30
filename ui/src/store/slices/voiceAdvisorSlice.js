import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  enabled: false,
  loading: false,
  error: null,
  lastMessage: null,
  settings: {
    volume: 1,
    rate: 1,
    pitch: 1,
    language: 'en-US',
  },
};

const slice = createSlice({
  name: 'voiceAdvisor',
  initialState,
  reducers: {
    setEnabled(state, action) {
      state.enabled = action.payload;
    },
    setLoading(state, action) {
      state.loading = action.payload;
    },
    setError(state, action) {
      state.error = action.payload;
    },
    setLastMessage(state, action) {
      state.lastMessage = action.payload;
    },
    updateSettings(state, action) {
      state.settings = { ...state.settings, ...action.payload };
    },
  },
});

export const { actions } = slice;
export default slice;
