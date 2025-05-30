import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Box, Typography, Paper, Switch } from '@mui/material';
import { api, socketManager } from '../api';
import { actions as voiceActions } from '../store/slices/voiceAdvisorSlice';

const VoiceAdvisorPanel = () => {
  const dispatch = useDispatch();
  const { enabled, lastMessage } = useSelector((state) => state.voiceAdvisor);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const { data } = await api.voiceAdvisor.getStatus();
        dispatch(voiceActions.setEnabled(data.enabled));
      } catch (err) {
        console.error('Failed to load voice advisor status', err);
      }
    };

    fetchStatus();
    const unsub = socketManager.subscribe('voice-advisor:message', (msg) => {
      dispatch(voiceActions.setLastMessage(msg));
    });
    return () => unsub();
  }, [dispatch]);

  const toggle = async () => {
    try {
      if (enabled) {
        await api.voiceAdvisor.disable();
      } else {
        await api.voiceAdvisor.enable();
      }
      dispatch(voiceActions.setEnabled(!enabled));
    } catch (err) {
      console.error('Failed to toggle voice advisor', err);
    }
  };

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
          Voice Advisor
        </Typography>
        <Switch checked={enabled} onChange={toggle} />
      </Box>
      {lastMessage && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          {lastMessage.message}
        </Typography>
      )}
    </Paper>
  );
};

export default VoiceAdvisorPanel;
