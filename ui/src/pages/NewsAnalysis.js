import React from 'react';
import { Box, Typography } from '@mui/material';
import NewsFeed from '../components/NewsFeed/NewsFeed';

const NewsAnalysis = () => (
  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
    <Typography variant="h4">News Analysis</Typography>
    <NewsFeed />
  </Box>
);

export default NewsAnalysis;
