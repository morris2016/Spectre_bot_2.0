import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  MenuItem,
  Button,
  Paper,
  CircularProgress,
  Grid
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { api } from '../api';
import { actions as brainActions } from '../store/slices/brainSlice';

const STRATEGY_TYPES = [
  { label: 'Reinforcement Learning', value: 'reinforcement' },
  { label: 'Rule Based', value: 'rule_based' }
];

const StrategyBuilder = () => {
  const dispatch = useDispatch();
  const brains = useSelector((state) => state.brain.brains);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    name: '',
    type: 'reinforcement',
    description: '',
    parameters: ''
  });

  useEffect(() => {
    const loadBrains = async () => {
      try {
        const { data } = await api.brain.getBrains();
        dispatch(brainActions.setBrains(data || []));
      } catch (err) {
        console.error('Failed to load brains', err);
      }
    };
    loadBrains();
  }, [dispatch]);

  const handleChange = (field) => (event) => {
    setForm({ ...form, [field]: event.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const { data } = await api.brain.createBrain({
        name: form.name,
        type: form.type,
        description: form.description,
        parameters: form.parameters
      });
      if (data) {
        dispatch(brainActions.addBrain(data));
        setForm({ name: '', type: 'reinforcement', description: '', parameters: '' });
      }
    } catch (err) {
      console.error('Failed to create brain', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto', mt: 2 }}>
      <Typography variant="h4" gutterBottom>
        Strategy Builder
      </Typography>
      <Paper sx={{ p: 3 }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Name"
                value={form.name}
                onChange={handleChange('name')}
                fullWidth
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                select
                label="Type"
                value={form.type}
                onChange={handleChange('type')}
                fullWidth
              >
                {STRATEGY_TYPES.map((opt) => (
                  <MenuItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Description"
                value={form.description}
                onChange={handleChange('description')}
                fullWidth
                multiline
                rows={3}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Parameters (JSON)"
                value={form.parameters}
                onChange={handleChange('parameters')}
                fullWidth
                multiline
                rows={4}
              />
            </Grid>
            <Grid item xs={12} sx={{ textAlign: 'right' }}>
              <Button type="submit" variant="contained" disabled={loading}>
                {loading ? <CircularProgress size={24} /> : 'Create Strategy'}
              </Button>
            </Grid>
          </Grid>
        </form>
      </Paper>

      {brains.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            Existing Strategies
          </Typography>
          <ul>
            {brains.map((brain) => (
              <li key={brain.id}>{brain.name} - {brain.type}</li>
            ))}
          </ul>
        </Box>
      )}
    </Box>
  );
};

export default StrategyBuilder;
