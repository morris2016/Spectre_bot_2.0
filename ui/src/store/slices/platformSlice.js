import { createSlice } from '@reduxjs/toolkit';

const slice = createSlice({
  name: 'platform',
  initialState: {
    selected: null,
    available: []
  },
  reducers: {
    setPlatforms(state, action) {
      state.available = action.payload;
    },
    selectPlatform(state, action) {
      state.selected = action.payload;
    }
  }
});

export const { actions } = slice;
export default slice;
