import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { api, socketManager } from '../api';

/**
 * Analytics page displaying key performance metrics with live updates.
 */
const Analytics = () => {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadMetrics = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data } = await api.system.getMetrics({ limit: 100 });
        setMetrics(data || []);
      } catch (err) {
        console.error('Failed to load metrics', err);
        setError('Failed to load metrics');
      } finally {
        setLoading(false);
      }
    };

    loadMetrics();
  }, []);

  useEffect(() => {
    const unsub = socketManager.subscribe('system:metrics', (data) => {
      setMetrics((prev) => [...prev.slice(-99), data]);
    });
    return () => unsub();
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">System Analytics</Typography>
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
          <CircularProgress />
        </Box>
      )}
      {error && (
        <Typography color="error" variant="body2" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}
      {metrics.length > 0 && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            CPU Usage
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <XAxis dataKey="timestamp" tick={false} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
              <Tooltip formatter={(v) => `${v}%`} labelFormatter={() => ''} />
              <Line type="monotone" dataKey="cpu" stroke="#8884d8" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      )}
    </Box>
  );
};

export default Analytics;
