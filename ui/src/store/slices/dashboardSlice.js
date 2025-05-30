import { createSlice } from '@reduxjs/toolkit';
import { api } from '../../api';

const initialState = {
  data: null,
  loading: false,
  error: null,
};

const slice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    startLoading(state) {
      state.loading = true;
      state.error = null;
    },
    fetchSuccess(state, action) {
      state.loading = false;
      state.data = action.payload;
    },
    fetchFailure(state, action) {
      state.loading = false;
      state.error = action.payload;
    },
  },
});

export const loadDashboard = () => async (dispatch) => {
  dispatch(actions.startLoading());
  try {
    const [systemRes, portfolioRes] = await Promise.all([
      api.system.getStatus(),
      api.portfolio.getSummary(),
    ]);
    dispatch(
      actions.fetchSuccess({
        system: systemRes.data,
        portfolio: portfolioRes.data,
      }),
    );
  } catch (err) {
    const message = err.response?.data?.message || err.message;
    dispatch(actions.fetchFailure(message));
  }
};

export const { actions } = slice;
export default slice;
