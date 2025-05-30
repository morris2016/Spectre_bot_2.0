import React from 'react';
import { Box, Typography } from '@mui/material';
import TradingView from '../components/TradingView/TradingView';
import MarketSentimentIndicator from '../components/MarketSentimentIndicator';
import { useSelector } from 'react-redux';
import { selectCurrentAsset } from '../store/slices/assetSlice';

const MarketAnalysis = () => {
  const currentAsset = useSelector(selectCurrentAsset);
  const symbol = currentAsset ? currentAsset.symbol : null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Market Analysis</Typography>
      <Box sx={{ height: 500 }}>
        <TradingView />
      </Box>
      {symbol && (
        <MarketSentimentIndicator symbol={symbol} />
      )}
    </Box>
  );
};

export default MarketAnalysis;
