import React, { useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  CircularProgress,
} from '@mui/material';
import { api } from '../api';

/**
 * Display authenticated user details and portfolio balances.
 */
const AccountDetails = () => {
  const [profile, setProfile] = useState(null);
  const [balances, setBalances] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [profileRes, balanceRes] = await Promise.all([
          api.auth.getProfile(),
          api.portfolio.getBalance(),
        ]);
        setProfile(profileRes.data);
        setBalances(balanceRes.data || []);
      } catch (err) {
        console.error('Failed to load account details', err);
        setError('Failed to load account details');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Account Details</Typography>
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
      {profile && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Profile</Typography>
          <Typography variant="body2">Name: {profile.name}</Typography>
          <Typography variant="body2">Email: {profile.email}</Typography>
        </Paper>
      )}
      {balances.length > 0 && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Balances</Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Asset</TableCell>
                <TableCell align="right">Free</TableCell>
                <TableCell align="right">Locked</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {balances.map((b) => (
                <TableRow key={b.asset}>
                  <TableCell>{b.asset}</TableCell>
                  <TableCell align="right">{b.free}</TableCell>
                  <TableCell align="right">{b.locked}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>
      )}
    </Box>
  );
};

export default AccountDetails;
