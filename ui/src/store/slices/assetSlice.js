import { createSlice } from '@reduxjs/toolkit';
import { api } from '../../api';

const initialState = {
  items: [],
  current: null,
  loading: false,
  error: null,
};

const slice = createSlice({
  name: 'asset',
  initialState,
  reducers: {
    startLoading(state) {
      state.loading = true;
      state.error = null;
    },
    fetchSuccess(state, action) {
      state.loading = false;
      state.items = action.payload;
    },
    fetchFailure(state, action) {
      state.loading = false;
      state.error = action.payload;
    },
    setCurrentAsset(state, action) {
      state.current = action.payload;
    },
  },
});

export const fetchAssets = (platformId) => async (dispatch) => {
  dispatch(actions.startLoading());
  try {
    const { data } = await api.market.getSymbols(platformId);
    dispatch(actions.fetchSuccess(data));
  } catch (err) {
    const message = err.response?.data?.message || err.message;
    dispatch(actions.fetchFailure(message));
  }
};

export const { actions } = slice;
export default slice;
