import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
} from '@mui/material';
import { actions as platformActions } from '../store/slices/platformSlice';
import { fetchAssets } from '../store/slices/assetSlice';
import { api } from '../api';

const PlatformSwitcher = () => {
  const dispatch = useDispatch();
  const platforms = useSelector((state) => state.platform.available);
  const selected = useSelector((state) => state.platform.selected);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [platformId, setPlatformId] = useState(selected || '');
  const [apiKey, setApiKey] = useState('');
  const [secret, setSecret] = useState('');

  useEffect(() => {
    const loadPlatforms = async () => {
      if (!platforms.length) {
        try {
          const { data } = await api.platform.getPlatforms();
          dispatch(platformActions.setPlatforms(data));
        } catch (err) {
          console.error(err);
        }
      }
    };
    loadPlatforms();
  }, [dispatch, platforms.length]);

  useEffect(() => {
    if (selected) {
      dispatch(fetchAssets(selected));
    }
  }, [dispatch, selected]);

  const handleSelect = (event) => {
    setPlatformId(event.target.value);
    setDialogOpen(true);
  };

  const handleSave = async () => {
    try {
      await api.platform.updateCredentials(platformId, { apiKey, secret });
      dispatch(platformActions.selectPlatform(platformId));
      dispatch(fetchAssets(platformId));
    } catch (err) {
      console.error(err);
    }
    setDialogOpen(false);
    setApiKey('');
    setSecret('');
  };

  return (
    <Box sx={{ p: 2 }}>
      <FormControl size="small" sx={{ minWidth: 160 }}>
        <InputLabel id="platform-select-label">Platform</InputLabel>
        <Select
          labelId="platform-select-label"
          label="Platform"
          value={selected || ''}
          onChange={handleSelect}
        >
          {platforms.map((p) => (
            <MenuItem key={p.id} value={p.id}>
              {p.name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)}>
        <DialogTitle>Enter API Credentials</DialogTitle>
        <DialogContent>
          <TextField
            margin="dense"
            label="API Key"
            fullWidth
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <TextField
            margin="dense"
            label="Secret"
            type="password"
            fullWidth
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSave} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PlatformSwitcher;
