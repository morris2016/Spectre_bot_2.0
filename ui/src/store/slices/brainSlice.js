import { createSlice } from '@reduxjs/toolkit';

const slice = createSlice({
  name: 'brain',
  initialState: {
    status: 'idle',
    activeBrainId: null,
    brains: []
  },
  reducers: {
    setStatus(state, action) {
      state.status = action.payload;
    },
    setActiveBrain(state, action) {
      state.activeBrainId = action.payload;
    },
    setBrains(state, action) {
      state.brains = action.payload;
    },
    addBrain(state, action) {
      state.brains.push(action.payload);
    },
    removeBrain(state, action) {
      state.brains = state.brains.filter((b) => b.id !== action.payload);
    }
  }
});

export const { actions } = slice;
export default slice;
