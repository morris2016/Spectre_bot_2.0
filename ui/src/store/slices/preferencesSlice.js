import { createSlice } from '@reduxjs/toolkit';
import { api } from '../../api';

export const initializePreferences = () => async (dispatch) => {
  dispatch(actions.startLoading());
  try {
    const { data } = await api.settings.getAll();
    dispatch(actions.loadSuccess(data));
  } catch (err) {
    const message = err.response?.data?.message || err.message;
    dispatch(actions.loadFailure(message));
  }
};

const slice = createSlice({
  name: 'preferences',
  initialState: { data: null, loading: false, error: null },
  reducers: {
    startLoading(state) {
      state.loading = true;
      state.error = null;
    },
    loadSuccess(state, action) {
      state.loading = false;
      state.data = action.payload;
    },
    loadFailure(state, action) {
      state.loading = false;
      state.error = action.payload;
    },
  }
});

export const { actions } = slice;
export default slice;
