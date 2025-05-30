import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  CircularProgress,
  Breadcrumbs,
  Link,
  Divider,
  Grid,
  Tabs,
  Tab,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Psychology,
  Home,
  NavigateNext,
  TrendingUp,
  Article,
  School,
  Memory,
  Refresh,
  Settings,
  BarChart,
  ShowChart,
  CloudUpload
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import CouncilDashboard from '../components/BrainCouncil/CouncilDashboard';
import PredictiveChart from '../components/LiveTrading/PredictiveChart';
import NewsFeedPanel from '../components/LiveTrading/NewsFeedPanel';
import VoiceAssistantPanel from '../components/VoiceAssistant/VoiceAssistantPanel';
import DocumentUploader from '../components/Learning/DocumentUploader';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import api from '../api';

const EnhancedIntelligence = () => {
  const theme = useTheme();
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedAsset, setSelectedAsset] = useState('BTCUSDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [selectedPlatform, setSelectedPlatform] = useState('binance');
  const [strategyLabTab, setStrategyLabTab] = useState(0);
  const [availableAssets, setAvailableAssets] = useState([
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT'
  ]);
  const [availableTimeframes, setAvailableTimeframes] = useState([
    '1m', '5m', '15m', '30m', '1h', '4h', '1d'
  ]);
  const [availablePlatforms, setAvailablePlatforms] = useState([
    'binance', 'deriv'
  ]);
  const [latestPrediction, setLatestPrediction] = useState(null);

  useEffect(() => {
    // Check if user is authenticated
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }

    // Fetch available assets and timeframes
    const fetchAvailableOptions = async () => {
      try {
        const response = await api.get('/api/market/available-options');
        if (response.data) {
          if (response.data.assets) {
            setAvailableAssets(response.data.assets);
            if (response.data.assets.length > 0) {
              setSelectedAsset(response.data.assets[0]);
            }
          }
          
          if (response.data.timeframes) {
            setAvailableTimeframes(response.data.timeframes);
            if (response.data.timeframes.includes('1h')) {
              setSelectedTimeframe('1h');
            } else if (response.data.timeframes.length > 0) {
              setSelectedTimeframe(response.data.timeframes[0]);
            }
          }
          
          if (response.data.platforms) {
            setAvailablePlatforms(response.data.platforms);
            if (response.data.platforms.length > 0) {
              setSelectedPlatform(response.data.platforms[0]);
            }
          }
        }
      } catch (err) {
        console.error('Error fetching available options:', err);
      }
    };

    fetchAvailableOptions();
    
    // Simulate loading
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, [isAuthenticated, navigate]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleAssetChange = (event) => {
    setSelectedAsset(event.target.value);
  };

  const handleTimeframeChange = (event) => {
    setSelectedTimeframe(event.target.value);
  };

  const handlePlatformChange = (event) => {
    setSelectedPlatform(event.target.value);
  };

  const handlePredictionUpdate = (prediction) => {
    setLatestPrediction(prediction);
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="80vh"
        >
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box mt={3} mb={4}>
        <Breadcrumbs
          separator={<NavigateNext fontSize="small" />}
          aria-label="breadcrumb"
        >
          <Link
            color="inherit"
            href="/"
            onClick={(e) => {
              e.preventDefault();
              navigate('/');
            }}
            style={{ display: 'flex', alignItems: 'center' }}
          >
            <Home fontSize="small" style={{ marginRight: theme.spacing(0.5) }} />
            Home
          </Link>
          <Typography
            color="textPrimary"
            style={{ display: 'flex', alignItems: 'center' }}
          >
            <Psychology fontSize="small" style={{ marginRight: theme.spacing(0.5) }} />
            Enhanced Intelligence
          </Typography>
        </Breadcrumbs>

        <Box mt={3} mb={3}>
          <Paper
            elevation={0}
            style={{
              padding: theme.spacing(3),
              backgroundColor: theme.palette.primary.dark,
              color: theme.palette.primary.contrastText,
              borderRadius: theme.shape.borderRadius * 2
            }}
          >
            <Typography variant="h4" component="h1" gutterBottom>
              Enhanced Intelligence System
            </Typography>
            <Typography variant="subtitle1">
              Advanced decision-making with asset-specific councils and ML model integration
            </Typography>
          </Paper>
        </Box>

        <Paper>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            variant="fullWidth"
          >
            <Tab icon={<Memory />} label="Brain Council" />
            <Tab icon={<ShowChart />} label="Live Trading" />
            <Tab icon={<Psychology />} label="Voice Assistant" />
            <Tab icon={<CloudUpload />} label="Document Learning" />
          </Tabs>
        </Paper>

        <Box mt={3}>
          {activeTab === 0 && (
            <CouncilDashboard />
          )}
          
          {activeTab === 1 && (
            <>
              <Box mb={3} display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="h6">Live Trading Intelligence</Typography>
                
                <Box display="flex" alignItems="center">
                  <FormControl variant="outlined" size="small" style={{ minWidth: 120, marginRight: theme.spacing(2) }}>
                    <InputLabel id="platform-select-label">Platform</InputLabel>
                    <Select
                      labelId="platform-select-label"
                      value={selectedPlatform}
                      onChange={handlePlatformChange}
                      label="Platform"
                    >
                      {availablePlatforms.map((platform) => (
                        <MenuItem key={platform} value={platform} style={{ textTransform: 'capitalize' }}>
                          {platform}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <FormControl variant="outlined" size="small" style={{ minWidth: 120, marginRight: theme.spacing(2) }}>
                    <InputLabel id="asset-select-label">Asset</InputLabel>
                    <Select
                      labelId="asset-select-label"
                      value={selectedAsset}
                      onChange={handleAssetChange}
                      label="Asset"
                    >
                      {availableAssets.map((asset) => (
                        <MenuItem key={asset} value={asset}>
                          {asset}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <FormControl variant="outlined" size="small" style={{ minWidth: 120 }}>
                    <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
                    <Select
                      labelId="timeframe-select-label"
                      value={selectedTimeframe}
                      onChange={handleTimeframeChange}
                      label="Timeframe"
                    >
                      {availableTimeframes.map((tf) => (
                        <MenuItem key={tf} value={tf}>
                          {tf}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>
              </Box>
              
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <PredictiveChart
                    asset={selectedAsset}
                    platform={selectedPlatform}
                    timeframe={selectedTimeframe}
                    onPredictionUpdate={handlePredictionUpdate}
                  />
                </Grid>
                
                <Grid item xs={12} md={8}>
                  <NewsFeedPanel asset={selectedAsset} />
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <VoiceAssistantPanel />
                </Grid>
              </Grid>
            </>
          )}
          
          {activeTab === 2 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <VoiceAssistantPanel />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Paper elevation={0} variant="outlined" style={{ padding: theme.spacing(2) }}>
                  <Typography variant="h6" gutterBottom>Voice Assistant Help</Typography>
                  <Typography variant="body2" paragraph>
                    The voice assistant can help you with:
                  </Typography>
                  
                  <Box mb={2}>
                    <Typography variant="subtitle2" gutterBottom>Trading Commands</Typography>
                    <ul style={{ paddingLeft: theme.spacing(2) }}>
                      <li>Show me the price prediction for [asset]</li>
                      <li>What's the market sentiment for [asset]?</li>
                      <li>Place a buy order for [asset]</li>
                      <li>What's my portfolio performance?</li>
                    </ul>
                  </Box>
                  
                  <Box mb={2}>
                    <Typography variant="subtitle2" gutterBottom>Analysis Commands</Typography>
                    <ul style={{ paddingLeft: theme.spacing(2) }}>
                      <li>Analyze the current market regime</li>
                      <li>Show me support and resistance levels</li>
                      <li>What patterns do you see in [asset]?</li>
                      <li>Compare [asset1] and [asset2]</li>
                    </ul>
                  </Box>
                  
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>System Commands</Typography>
                    <ul style={{ paddingLeft: theme.spacing(2) }}>
                      <li>Switch to [timeframe] timeframe</li>
                      <li>Change asset to [asset]</li>
                      <li>Show me the brain council dashboard</li>
                      <li>Explain how the ML models work</li>
                    </ul>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          )}
          
          {activeTab === 3 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <DocumentUploader />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Paper elevation={0} variant="outlined" style={{ padding: theme.spacing(2) }}>
                  <Typography variant="h6" gutterBottom>Document Learning</Typography>
                  <Typography variant="body2" paragraph>
                    Upload documents to teach the AI new trading strategies, analysis techniques, and market knowledge.
                  </Typography>
                  
                  <Box mb={2}>
                    <Typography variant="subtitle2" gutterBottom>Supported Document Types</Typography>
                    <ul style={{ paddingLeft: theme.spacing(2) }}>
                      <li>Trading strategies and systems</li>
                      <li>Market analysis reports</li>
                      <li>Research papers</li>
                      <li>Trading books and guides</li>
                      <li>Code examples (Python, JavaScript)</li>
                    </ul>
                  </Box>
                  
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>How It Works</Typography>
                    <Typography variant="body2">
                      The AI will analyze your documents, extract key insights, and integrate the knowledge into its decision-making process. This allows the system to continuously learn and improve its trading strategies based on your preferred approaches.
                    </Typography>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          )}
          
          {activeTab === 4 && (
            <>
              <Box mb={3}>
                <Tabs
                  value={strategyLabTab}
                  onChange={(e, newValue) => setStrategyLabTab(newValue)}
                  indicatorColor="secondary"
                  textColor="secondary"
                  variant="fullWidth"
                >
                  <Tab icon={<TrendingUp />} label="Strategy Simulator" />
                  <Tab icon={<School />} label="Knowledge Base" />
                </Tabs>
              </Box>
              
              {strategyLabTab === 0 && (
                <Box mb={3}>
                  <Typography variant="h6" gutterBottom>Advanced Strategy Simulation</Typography>
                  <Typography variant="body2" color="textSecondary" paragraph>
                    Test and optimize trading strategies with historical data and self-learning capabilities
                  </Typography>
                  
                  <Grid container spacing={3}>
                    <Grid item xs={12}>
                      <StrategySimulator />
                    </Grid>
                  </Grid>
                </Box>
              )}
              
              {strategyLabTab === 1 && (
                <Box mb={3}>
                  <Typography variant="h6" gutterBottom>Trading Knowledge</Typography>
                  <Typography variant="body2" color="textSecondary" paragraph>
                    Explore advanced trading concepts and methodologies
                  </Typography>
                  
                  <Grid container spacing={3}>
                    <Grid item xs={12}>
                      <KnowledgeBase />
                    </Grid>
                  </Grid>
                </Box>
              )}
            </>
          )}
        </Box>
      </Box>
    </Container>
  );
};

export default EnhancedIntelligence;