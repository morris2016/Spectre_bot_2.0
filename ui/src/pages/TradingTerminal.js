import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  CircularProgress,
} from '@mui/material';
import TradingView from '../components/TradingView/TradingView';
import OrderPanel from '../components/OrderPanel/OrderPanel';
import { api, socketManager } from '../api';

/**
 * Full trading terminal with chart, positions and order management.
 */
const TradingTerminal = () => {
  const [openOrders, setOpenOrders] = useState([]);
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [ordersRes, positionsRes] = await Promise.all([
          api.trading.getOpenOrders({}),
          api.trading.getPositions({}),
        ]);
        setOpenOrders(ordersRes.data || []);
        setPositions(positionsRes.data || []);
      } catch (err) {
        console.error('Failed to load trading terminal data', err);
        setError('Failed to load trading terminal data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  useEffect(() => {
    const unsubOrders = socketManager.subscribe('orders:update', (data) => {
      setOpenOrders(data);
    });
    const unsubPositions = socketManager.subscribe('positions:update', (data) => {
      setPositions(data);
    });
    return () => {
      unsubOrders();
      unsubPositions();
    };
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Trading Terminal</Typography>
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
      <Box sx={{ height: 500 }}>
        <TradingView />
      </Box>
      <OrderPanel />

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6">Open Positions</Typography>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Quantity</TableCell>
              <TableCell>Entry</TableCell>
              <TableCell>PnL</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {positions.map((p) => (
              <TableRow key={p.id}>
                <TableCell>{p.symbol}</TableCell>
                <TableCell>{p.quantity}</TableCell>
                <TableCell>{p.entryPrice}</TableCell>
                <TableCell>{p.unrealizedPnl}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6">Open Orders</Typography>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Symbol</TableCell>
              <TableCell>Side</TableCell>
              <TableCell>Price</TableCell>
              <TableCell>Quantity</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {openOrders.map((o) => (
              <TableRow key={o.id}>
                <TableCell>{o.id}</TableCell>
                <TableCell>{o.symbol}</TableCell>
                <TableCell>{o.side}</TableCell>
                <TableCell>{o.price}</TableCell>
                <TableCell>{o.quantity}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
    </Box>
  );
};

export default TradingTerminal;
