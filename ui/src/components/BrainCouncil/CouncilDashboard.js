import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Grid, 
  Typography, 
  Box, 
  Paper, 
  Tabs, 
  Tab, 
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  IconButton,
  Tooltip,
  Card,
  CardHeader,
  CardContent,
  CircularProgress
} from '@mui/material';
import { 
  Psychology, 
  Timeline, 
  Insights, 
  Refresh,
  ExpandMore,
  ExpandLess,
  Dashboard,
  BubbleChart,
  Memory
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import AssetCouncilPanel from './AssetCouncilPanel';
import MLCouncilPanel from './MLCouncilPanel';
import api from '../../api';

const CouncilDashboard = () => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedAssets, setSelectedAssets] = useState([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [availableAssets, setAvailableAssets] = useState([]);
  const [availableTimeframes, setAvailableTimeframes] = useState([]);
  const [expanded, setExpanded] = useState(false);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/brain-council/dashboard');
      setDashboardData(response.data);
      
      // Set available assets and timeframes
      if (response.data.available_assets) {
        setAvailableAssets(response.data.available_assets);
        // Select first 4 assets by default
        setSelectedAssets(response.data.available_assets.slice(0, 4));
      }
      
      if (response.data.available_timeframes) {
        setAvailableTimeframes(response.data.available_timeframes);
        // Select 1h timeframe by default if available
        if (response.data.available_timeframes.includes('1h')) {
          setSelectedTimeframe('1h');
        } else if (response.data.available_timeframes.length > 0) {
          setSelectedTimeframe(response.data.available_timeframes[0]);
        }
      }
      
      setError(null);
    } catch (err) {
      console.error('Error fetching council dashboard data:', err);
      setError('Failed to load council dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const handleTimeframeChange = (event) => {
    setSelectedTimeframe(event.target.value);
  };

  const handleAssetToggle = (asset) => {
    if (selectedAssets.includes(asset)) {
      setSelectedAssets(selectedAssets.filter(a => a !== asset));
    } else {
      setSelectedAssets([...selectedAssets, asset]);
    }
  };

  const renderOverview = () => {
    if (!dashboardData) return null;
    
    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader 
              title="Brain Council System" 
              subheader="Enhanced Intelligence Architecture"
            />
            <CardContent>
              <Typography variant="body1" paragraph>
                The Brain Council System coordinates multiple specialized councils to make intelligent trading decisions.
              </Typography>
              
              <Box mt={2}>
                <Typography variant="subtitle1" gutterBottom>System Architecture</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Paper 
                      variant="outlined" 
                      style={{ 
                        padding: theme.spacing(2),
                        backgroundColor: theme.palette.primary.light,
                        color: theme.palette.primary.contrastText
                      }}
                    >
                      <Box display="flex" alignItems="center" mb={1}>
                        <Dashboard style={{ marginRight: theme.spacing(1) }} />
                        <Typography variant="subtitle2">Council Manager</Typography>
                      </Box>
                      <Typography variant="body2">
                        Coordinates all councils and provides a unified decision-making interface
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper 
                      variant="outlined" 
                      style={{ 
                        padding: theme.spacing(2),
                        backgroundColor: theme.palette.secondary.light,
                        color: theme.palette.secondary.contrastText
                      }}
                    >
                      <Box display="flex" alignItems="center" mb={1}>
                        <BubbleChart style={{ marginRight: theme.spacing(1) }} />
                        <Typography variant="subtitle2">Asset Councils</Typography>
                      </Box>
                      <Typography variant="body2">
                        Specialized councils for each trading asset
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper 
                      variant="outlined" 
                      style={{ 
                        padding: theme.spacing(2),
                        backgroundColor: theme.palette.success.light,
                        color: theme.palette.success.contrastText
                      }}
                    >
                      <Box display="flex" alignItems="center" mb={1}>
                        <Memory style={{ marginRight: theme.spacing(1) }} />
                        <Typography variant="subtitle2">ML Council</Typography>
                      </Box>
                      <Typography variant="body2">
                        Integrates machine learning models into the decision process
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
              
              <Box mt={3}>
                <Typography variant="subtitle1" gutterBottom>System Statistics</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Box p={1} textAlign="center">
                      <Typography variant="h4" color="primary">
                        {dashboardData.asset_count || 0}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Active Assets
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box p={1} textAlign="center">
                      <Typography variant="h4" color="secondary">
                        {dashboardData.brain_count || 0}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Strategy Brains
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box p={1} textAlign="center">
                      <Typography variant="h4" color="success.main">
                        {dashboardData.ml_model_count || 0}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        ML Models
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box p={1} textAlign="center">
                      <Typography variant="h4" color="info.main">
                        {dashboardData.signals_per_hour || 0}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Signals/Hour
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <MLCouncilPanel />
        </Grid>
      </Grid>
    );
  };

  const renderAssetCouncils = () => {
    return (
      <Box>
        <Box mb={3} display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography variant="h6" gutterBottom>Asset Councils</Typography>
            <Typography variant="body2" color="textSecondary">
              Specialized decision-making for individual trading assets
            </Typography>
          </Box>
          
          <FormControl variant="outlined" size="small" style={{ minWidth: 120 }}>
            <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
            <Select
              labelId="timeframe-select-label"
              id="timeframe-select"
              value={selectedTimeframe}
              onChange={handleTimeframeChange}
              label="Timeframe"
            >
              {availableTimeframes.map((tf) => (
                <MenuItem key={tf} value={tf}>{tf}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        
        <Box mb={3}>
          <Paper variant="outlined" style={{ padding: theme.spacing(1) }}>
            <Typography variant="subtitle2" gutterBottom>Select Assets</Typography>
            <Box display="flex" flexWrap="wrap">
              {availableAssets.map((asset) => (
                <Button
                  key={asset}
                  variant={selectedAssets.includes(asset) ? "contained" : "outlined"}
                  size="small"
                  color={selectedAssets.includes(asset) ? "primary" : "default"}
                  onClick={() => handleAssetToggle(asset)}
                  style={{ margin: theme.spacing(0.5) }}
                >
                  {asset}
                </Button>
              ))}
            </Box>
          </Paper>
        </Box>
        
        <Grid container spacing={3}>
          {selectedAssets.map((asset) => (
            <Grid item xs={12} md={6} lg={4} key={asset}>
              <AssetCouncilPanel assetId={asset} timeframe={selectedTimeframe} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  if (loading && !dashboardData) {
    return (
      <Container maxWidth="lg">
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg">
        <Box mt={4} textAlign="center">
          <Typography color="error" variant="h6" gutterBottom>{error}</Typography>
          <Button 
            variant="contained" 
            color="primary" 
            startIcon={<Refresh />} 
            onClick={fetchDashboardData}
          >
            Retry
          </Button>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box mt={4} mb={4}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
          <Box display="flex" alignItems="center">
            <Psychology fontSize="large" color="primary" style={{ marginRight: theme.spacing(2) }} />
            <Typography variant="h4" component="h1">
              Enhanced Intelligence System
            </Typography>
          </Box>
          <Button 
            variant="outlined" 
            startIcon={<Refresh />} 
            onClick={fetchDashboardData}
          >
            Refresh
          </Button>
        </Box>
        
        <Paper>
          <Tabs
            value={selectedTab}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            variant="fullWidth"
          >
            <Tab label="Overview" icon={<Insights />} />
            <Tab label="Asset Councils" icon={<BubbleChart />} />
          </Tabs>
        </Paper>
        
        <Box mt={3}>
          {selectedTab === 0 && renderOverview()}
          {selectedTab === 1 && renderAssetCouncils()}
        </Box>
      </Box>
    </Container>
  );
};

export default CouncilDashboard;