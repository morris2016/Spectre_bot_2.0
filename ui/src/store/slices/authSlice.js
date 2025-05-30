import { createSlice } from '@reduxjs/toolkit';
import { api } from '../../api';

const initialState = {
  isAuthenticated: false,
  user: null,
  token: localStorage.getItem('token'),
  loading: false,
  error: null,
};

const slice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    startLoading(state) {
      state.loading = true;
      state.error = null;
    },
    loginSuccess(state, action) {
      state.loading = false;
      state.isAuthenticated = true;
      state.user = action.payload.user;
      state.token = action.payload.token;
    },
    logoutSuccess(state) {
      state.isAuthenticated = false;
      state.user = null;
      state.token = null;
    },
    authFailure(state, action) {
      state.loading = false;
      state.error = action.payload;
    },
    refreshTokenSuccess(state, action) {
      state.token = action.payload.token;
      localStorage.setItem('token', action.payload.token);
      if (action.payload.refreshToken) {
        localStorage.setItem('refreshToken', action.payload.refreshToken);
      }
    },
  },
});

export const login = (credentials) => async (dispatch) => {
  dispatch(actions.startLoading());
  try {
    const { data } = await api.auth.login(credentials);
    localStorage.setItem('token', data.token);
    localStorage.setItem('refreshToken', data.refreshToken);
    dispatch(actions.loginSuccess(data));
  } catch (err) {
    const message = err.response?.data?.message || err.message;
    dispatch(actions.authFailure(message));
  }
};

export const checkAuthStatus = () => async (dispatch) => {
  dispatch(actions.startLoading());
  const token = localStorage.getItem('token');
  if (!token) {
    dispatch(actions.logoutSuccess());
    return;
  }
  try {
    const { data } = await api.auth.getProfile();
    dispatch(actions.loginSuccess({ user: data, token }));
  } catch (err) {
    dispatch(actions.logoutSuccess());
  }
};

export const logout = () => async (dispatch) => {
  try {
    await api.auth.logout();
  } catch (err) {
    // ignore logout errors
  }
  localStorage.removeItem('token');
  localStorage.removeItem('refreshToken');
  dispatch(actions.logoutSuccess());
};

export const { actions } = slice;
export default slice;
