import React, { useEffect, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useSnackbar } from 'notistack';
import { actions as alertsActions } from '../store/slices/alertsSlice';

/**
 * NotificationCenter displays alerts from Redux using snackbar toasts.
 * Alerts are removed from the store when dismissed.
 */
const NotificationCenter = () => {
  const alerts = useSelector((state) => state.alerts.list);
  const dispatch = useDispatch();
  const { enqueueSnackbar } = useSnackbar();

import { useSnackbar } from 'notistack';
import { useDispatch, useSelector } from 'react-redux';
import { actions as alertsActions } from '../store/slices/alertsSlice';

/**
 * Subscribe to Redux alerts and display them via snackbars.
 */
const NotificationCenter = () => {
  const dispatch = useDispatch();
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const alerts = useSelector((state) => state.alerts.list);
  const shown = useRef(new Set());

  useEffect(() => {
    alerts.forEach((alert) => {
      if (shown.current.has(alert.id)) return;
      shown.current.add(alert.id);
      enqueueSnackbar(alert.message, {
        variant: alert.type || 'info',
        autoHideDuration: alert.timeout || 5000,
        onClose: () => dispatch(alertsActions.removeAlert(alert.id)),
      });
    });
  }, [alerts, enqueueSnackbar, dispatch]);

  useEffect(() => () => shown.current.clear(), []);

import { useSelector, useDispatch } from 'react-redux';
import { useSnackbar } from 'notistack';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import { alertsActions } from '../store';

/**
 * NotificationCenter integrates with notistack to display alerts
 * stored in the Redux state as transient snackbars. It ensures that
 * each alert is shown once and removed when dismissed.
 */
const NotificationCenter = () => {
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const alerts = useSelector((state) => state.alerts.list);
  const dispatch = useDispatch();

  // Keep track of displayed alerts so they are not shown multiple times
  const displayed = useRef([]);

  useEffect(() => {
    alerts.forEach((alert) => {
      if (displayed.current.includes(alert.id)) return;
      enqueueSnackbar(alert.message, {
        variant: alert.severity || 'info',
        onClose: (_, reason) => {
          if (reason === 'clickaway') return;
          dispatch(alertsActions.removeAlert(alert.id));
        },
        onExited: () => {
          dispatch(alertsActions.removeAlert(alert.id));
        },
      });
      displayed.current.push(alert.id);
    });
  }, [alerts, enqueueSnackbar, dispatch]);

  useEffect(() => {
    displayed.current = displayed.current.filter((id) =>
      alerts.some((alert) => alert.id === id)
    );
  }, [alerts]);

      enqueueSnackbar(alert.message, {
        variant: alert.type || 'info',
        autoHideDuration: alert.timeout || 5000,
        action: (key) => (
          <IconButton
            aria-label="close"
            color="inherit"
            size="small"
            onClick={() => {
              closeSnackbar(key);
              dispatch(alertsActions.removeAlert(alert.id));
            }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        ),
        onClose: () => dispatch(alertsActions.removeAlert(alert.id)),
      });

      displayed.current.push(alert.id);
    });
  }, [alerts, enqueueSnackbar, closeSnackbar, dispatch]);

  return null;
};

export default NotificationCenter;
