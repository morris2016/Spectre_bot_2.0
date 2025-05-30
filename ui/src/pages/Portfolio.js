import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Paper,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
} from '@mui/material';
import { api, socketManager } from '../api';

const Portfolio = () => {
  const [summary, setSummary] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [summaryRes, historyRes] = await Promise.all([
        api.portfolio.getSummary(),
        api.portfolio.getHistory({}),
      ]);
      setSummary(summaryRes.data);
      setHistory(historyRes.data || []);
    } catch (err) {
      console.error('Failed to load portfolio data', err);
      setError('Failed to load portfolio data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    const unsub = socketManager.subscribe('portfolio:update', (data) => {
      setSummary(data);
    });
    return () => unsub();
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Portfolio</Typography>
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
      {summary && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Balance</Typography>
          <Typography variant="body2">{summary.balance}</Typography>
        </Paper>
      )}
      {history.length > 0 && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Trade History</Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Date</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Side</TableCell>
                <TableCell>Price</TableCell>
                <TableCell>Quantity</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {history.map((h) => (
                <TableRow key={h.id}>
                  <TableCell>{h.date}</TableCell>
                  <TableCell>{h.symbol}</TableCell>
                  <TableCell>{h.side}</TableCell>
                  <TableCell>{h.price}</TableCell>
                  <TableCell>{h.quantity}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>
      )}
    </Box>
  );
};

export default Portfolio;
