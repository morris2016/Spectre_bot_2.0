import { createSlice } from '@reduxjs/toolkit';
import { api } from '../../api';

export const initializeSystem = () => async (dispatch) => {
  dispatch(actions.startLoading());
  try {
    const { data } = await api.system.getStatus();
    dispatch(actions.initializeSuccess(data));
  } catch (err) {
    const message = err.response?.data?.message || err.message;
    dispatch(actions.operationFailure(message));
  }
};

export const checkSystemHealth = () => async (dispatch) => {
  try {
    const { data } = await api.system.getStatus();
    dispatch(actions.updateHealth(data));
  } catch (err) {
    const message = err.response?.data?.message || err.message;
    dispatch(actions.operationFailure(message));
  }
};

const slice = createSlice({
  name: 'system',
  initialState: { initialized: false, loading: false, error: null, health: null },
  reducers: {
    startLoading(state) {
      state.loading = true;
      state.error = null;
    },
    initializeSuccess(state, action) {
      state.initialized = true;
      state.loading = false;
      state.health = action.payload;
    },
    updateHealth(state, action) {
      state.health = action.payload;
    },
    operationFailure(state, action) {
      state.loading = false;
      state.error = action.payload;
    },
  }
});

export const { actions } = slice;
export default slice;
