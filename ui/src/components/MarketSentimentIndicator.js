import React, { useEffect, useState } from 'react';
import { Box, Typography, CircularProgress, LinearProgress } from '@mui/material';
import PropTypes from 'prop-types';
import { api } from '../api';

const MarketSentimentIndicator = ({ symbol }) => {
  const [sentiment, setSentiment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!symbol) return;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data } = await api.market.getMarketSentiment(symbol);
        setSentiment(data);
      } catch (err) {
        console.error('Failed to fetch sentiment', err);
        setError('Failed to load sentiment');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [symbol]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CircularProgress size={16} />
        <Typography variant="body2">Loading sentiment...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Typography variant="body2" color="error">
        {error}
      </Typography>
    );
  }

  if (!sentiment) {
    return null;
  }

  const score = Math.max(0, Math.min(100, sentiment.score));
  return (
    <Box sx={{ width: 200 }}>
      <Typography variant="subtitle2" gutterBottom>
        Market Sentiment
      </Typography>
      <LinearProgress variant="determinate" value={score} />
      <Typography variant="caption">{score}% bullish</Typography>
    </Box>
  );
};

MarketSentimentIndicator.propTypes = {
  symbol: PropTypes.string.isRequired,
};

export default MarketSentimentIndicator;
