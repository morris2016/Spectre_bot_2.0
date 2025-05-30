import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Button, 
  Grid, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  TextField, 
  Slider, 
  Switch, 
  FormControlLabel, 
  Divider, 
  CircularProgress, 
  Chip, 
  Card,
  CardContent,
  CardHeader,
  Tabs,
  Tab
} from '@mui/material';
import { 
  PlayArrow, 
  Stop, 
  Save, 
  Refresh, 
  ExpandMore, 
  TrendingUp,
  TrendingDown,
  Psychology,
  Settings
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart as RechartsBarChart,
  Bar
} from 'recharts';
import api from '../../api';

// Preloaded strategy templates
const STRATEGY_TEMPLATES = [
  {
    id: 'wyckoff_accumulation',
    name: 'Wyckoff Accumulation',
    description: 'Identifies Wyckoff accumulation patterns for early trend detection',
    category: 'price_action',
    complexity: 'advanced',
    timeframes: ['1h', '4h', '1d'],
    parameters: {
      volume_threshold: 1.5,
      spring_detection: true,
      test_threshold: 0.8,
      confirmation_candles: 3
    },
    performance: {
      win_rate: 0.72,
      profit_factor: 2.1,
      max_drawdown: 0.15,
      avg_trade: 1.8
    }
  },
  {
    id: 'smart_money_concepts',
    name: 'Smart Money Concepts',
    description: 'Tracks institutional order flow and liquidity grabs',
    category: 'order_flow',
    complexity: 'advanced',
    timeframes: ['15m', '1h', '4h'],
    parameters: {
      liquidity_threshold: 2.0,
      imbalance_ratio: 0.7,
      mitigation_required: true,
      breaker_blocks: true
    },
    performance: {
      win_rate: 0.68,
      profit_factor: 2.3,
      max_drawdown: 0.18,
      avg_trade: 2.1
    }
  },
  {
    id: 'market_cipher',
    name: 'Market Cipher',
    description: 'Multi-indicator system with divergences and wave analysis',
    category: 'multi_indicator',
    complexity: 'advanced',
    timeframes: ['5m', '15m', '1h', '4h'],
    parameters: {
      rsi_length: 14,
      stoch_length: 9,
      ema_fast: 8,
      ema_slow: 21,
      divergence_threshold: 0.6
    },
    performance: {
      win_rate: 0.65,
      profit_factor: 1.9,
      max_drawdown: 0.22,
      avg_trade: 1.6
    }
  },
  {
    id: 'elliot_wave_oscillator',
    name: 'Elliott Wave Oscillator',
    description: 'Identifies Elliott Wave patterns with oscillator confirmation',
    category: 'wave_analysis',
    complexity: 'expert',
    timeframes: ['1h', '4h', '1d'],
    parameters: {
      wave_lookback: 100,
      oscillator_fast: 5,
      oscillator_slow: 35,
      fibonacci_levels: true
    },
    performance: {
      win_rate: 0.63,
      profit_factor: 2.4,
      max_drawdown: 0.25,
      avg_trade: 2.8
    }
  }
];

const StrategySimulator = () => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [simulationResults, setSimulationResults] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const [simulationSettings, setSimulationSettings] = useState({
    asset: 'BTCUSDT',
    timeframe: '1h',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
    riskPerTrade: 2,
    useRealData: true,
    enableSelfLearning: true,
    iterations: 1000
  });
  const [availableAssets] = useState([
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT'
  ]);
  const [strategyParameters, setStrategyParameters] = useState({});
  const [strategyTemplates] = useState(STRATEGY_TEMPLATES);

  // Update strategy parameters when a strategy is selected
  useEffect(() => {
    if (selectedStrategy) {
      setStrategyParameters(selectedStrategy.parameters);
    }
  }, [selectedStrategy]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleStrategySelect = (strategyId) => {
    const strategy = strategyTemplates.find(s => s.id === strategyId);
    setSelectedStrategy(strategy);
  };

  const handleSettingChange = (setting, value) => {
    setSimulationSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const handleParameterChange = (parameter, value) => {
    setStrategyParameters(prev => ({
      ...prev,
      [parameter]: value
    }));
  };

  const handleStartSimulation = async () => {
    if (!selectedStrategy) {
      return;
    }
    
    setIsSimulating(true);
    setSimulationProgress(0);
    
    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setSimulationProgress(prev => {
          const newProgress = prev + Math.random() * 5;
          return newProgress >= 100 ? 100 : newProgress;
        });
      }, 500);
      
      // Mock API call to run simulation
      setTimeout(() => {
        clearInterval(progressInterval);
        setSimulationProgress(100);
        
        // Mock results for demonstration
        setSimulationResults({
          summary: {
            total_trades: 124,
            winning_trades: 82,
            losing_trades: 42,
            win_rate: 0.66,
            profit_factor: 2.1,
            max_drawdown: 0.18,
            sharpe_ratio: 1.8,
            total_return: 0.87,
            annual_return: 0.42
          },
          equity_curve: Array(100).fill().map((_, i) => ({
            day: i,
            equity: 10000 * (1 + 0.003 * i + (Math.sin(i / 10) * 0.02))
          })),
          monthly_returns: [
            { month: 'Jan', return: 0.05 },
            { month: 'Feb', return: 0.03 },
            { month: 'Mar', return: -0.02 },
            { month: 'Apr', return: 0.07 },
            { month: 'May', return: 0.04 },
            { month: 'Jun', return: -0.01 },
            { month: 'Jul', return: 0.06 },
            { month: 'Aug', return: 0.08 },
            { month: 'Sep', return: -0.03 },
            { month: 'Oct', return: 0.05 },
            { month: 'Nov', return: 0.04 },
            { month: 'Dec', return: 0.06 }
          ],
          learning_insights: [
            "Identified optimal entry conditions during Asian trading session",
            "Improved performance by adjusting stop loss placement to account for volatility",
            "Discovered correlation between trade success and specific volume patterns",
            "Optimized take profit levels based on key resistance zones"
          ]
        });
        
        setIsSimulating(false);
      }, 3000);
    } catch (err) {
      console.error('Error running simulation:', err);
      setIsSimulating(false);
    }
  };

  const handleStopSimulation = () => {
    setIsSimulating(false);
  };

  const renderStrategyTemplates = () => {
    return (
      <Grid container spacing={2}>
        {strategyTemplates.map((template) => (
          <Grid item xs={12} md={6} key={template.id}>
            <Card 
              variant="outlined" 
              sx={{ 
                cursor: 'pointer',
                border: selectedStrategy?.id === template.id ? `2px solid ${theme.palette.primary.main}` : undefined,
                transition: 'all 0.2s',
                '&:hover': {
                  boxShadow: 3
                }
              }}
              onClick={() => handleStrategySelect(template.id)}
            >
              <CardHeader
                title={template.name}
                subheader={`Complexity: ${template.complexity}`}
                action={
                  <Chip 
                    label={template.category.replace('_', ' ')} 
                    size="small"
                    color="primary"
                    style={{ textTransform: 'capitalize' }}
                  />
                }
              />
              <CardContent>
                <Typography variant="body2" color="textSecondary" paragraph>
                  {template.description}
                </Typography>
                
                <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
                  <Box>
                    <Typography variant="caption" color="textSecondary">
                      Win Rate: {(template.performance.win_rate * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="caption" color="textSecondary" display="block">
                      Profit Factor: {template.performance.profit_factor.toFixed(1)}
                    </Typography>
                  </Box>
                  <Box>
                    <Chip 
                      size="small" 
                      label={template.timeframes.join(', ')}
                      variant="outlined"
                    />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };

  const renderSimulationSettings = () => {
    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>
            Simulation Settings
          </Typography>
          
          <FormControl fullWidth margin="normal">
            <InputLabel id="asset-select-label">Asset</InputLabel>
            <Select
              labelId="asset-select-label"
              value={simulationSettings.asset}
              onChange={(e) => handleSettingChange('asset', e.target.value)}
              label="Asset"
            >
              {availableAssets.map((asset) => (
                <MenuItem key={asset} value={asset}>{asset}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControl fullWidth margin="normal">
            <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
            <Select
              labelId="timeframe-select-label"
              value={simulationSettings.timeframe}
              onChange={(e) => handleSettingChange('timeframe', e.target.value)}
              label="Timeframe"
            >
              <MenuItem value="1m">1 Minute</MenuItem>
              <MenuItem value="5m">5 Minutes</MenuItem>
              <MenuItem value="15m">15 Minutes</MenuItem>
              <MenuItem value="30m">30 Minutes</MenuItem>
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="4h">4 Hours</MenuItem>
              <MenuItem value="1d">1 Day</MenuItem>
            </Select>
          </FormControl>
          
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                label="Start Date"
                type="date"
                fullWidth
                margin="normal"
                value={simulationSettings.startDate}
                onChange={(e) => handleSettingChange('startDate', e.target.value)}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                label="End Date"
                type="date"
                fullWidth
                margin="normal"
                value={simulationSettings.endDate}
                onChange={(e) => handleSettingChange('endDate', e.target.value)}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
          </Grid>
          
          <TextField
            label="Initial Capital"
            type="number"
            fullWidth
            margin="normal"
            value={simulationSettings.initialCapital}
            onChange={(e) => handleSettingChange('initialCapital', parseFloat(e.target.value))}
          />
          
          <Box mt={2}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Risk Per Trade (%)
            </Typography>
            <Slider
              value={simulationSettings.riskPerTrade}
              onChange={(e, newValue) => handleSettingChange('riskPerTrade', newValue)}
              min={0.1}
              max={10}
              step={0.1}
              valueLabelDisplay="auto"
            />
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>
            Advanced Settings
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={simulationSettings.useRealData}
                onChange={(e) => handleSettingChange('useRealData', e.target.checked)}
                color="primary"
              />
            }
            label="Use Real Historical Data"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={simulationSettings.enableSelfLearning}
                onChange={(e) => handleSettingChange('enableSelfLearning', e.target.checked)}
                color="primary"
              />
            }
            label="Enable Self-Learning"
          />
          
          <Box mt={2}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Simulation Iterations
            </Typography>
            <Slider
              value={simulationSettings.iterations}
              onChange={(e, newValue) => handleSettingChange('iterations', newValue)}
              min={100}
              max={10000}
              step={100}
              valueLabelDisplay="auto"
            />
          </Box>
          
          {selectedStrategy && (
            <Box mt={3}>
              <Typography variant="subtitle2" gutterBottom>
                Strategy Parameters
              </Typography>
              
              {Object.entries(strategyParameters).map(([param, value]) => (
                <Box key={param} mb={2}>
                  {typeof value === 'boolean' ? (
                    <FormControlLabel
                      control={
                        <Switch
                          checked={value}
                          onChange={(e) => handleParameterChange(param, e.target.checked)}
                          color="primary"
                        />
                      }
                      label={param.replace('_', ' ')}
                    />
                  ) : typeof value === 'number' ? (
                    <Box>
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        {param.replace('_', ' ')}
                      </Typography>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item xs>
                          <Slider
                            value={value}
                            onChange={(e, newValue) => handleParameterChange(param, newValue)}
                            min={0}
                            max={value * 2 || 10}
                            step={0.1}
                          />
                        </Grid>
                        <Grid item>
                          <TextField
                            value={value}
                            onChange={(e) => handleParameterChange(param, parseFloat(e.target.value))}
                            type="number"
                            size="small"
                            style={{ width: 80 }}
                          />
                        </Grid>
                      </Grid>
                    </Box>
                  ) : null}
                </Box>
              ))}
            </Box>
          )}
        </Grid>
      </Grid>
    );
  };

  const renderSimulationResults = () => {
    if (!simulationResults) {
      return (
        <Box textAlign="center" py={4}>
          <Typography variant="body1" color="textSecondary">
            Run a simulation to see results
          </Typography>
        </Box>
      );
    }
    
    return (
      <Box>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Performance Summary
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="body2" color="textSecondary">
                    Total Return
                  </Typography>
                  <Typography variant="h4" color={simulationResults.summary.total_return > 0 ? "success.main" : "error.main"}>
                    {(simulationResults.summary.total_return * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="body2" color="textSecondary">
                    Win Rate
                  </Typography>
                  <Typography variant="h4">
                    {(simulationResults.summary.win_rate * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="body2" color="textSecondary">
                    Profit Factor
                  </Typography>
                  <Typography variant="h4">
                    {simulationResults.summary.profit_factor.toFixed(2)}
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="body2" color="textSecondary">
                    Max Drawdown
                  </Typography>
                  <Typography variant="h4" color="error.main">
                    {(simulationResults.summary.max_drawdown * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
            
            <Box mt={3}>
              <Typography variant="subtitle2" gutterBottom>
                Monthly Returns
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <RechartsBarChart data={simulationResults.monthly_returns}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                  <RechartsTooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
                  <Bar 
                    dataKey="return" 
                    fill={(value) => value >= 0 ? theme.palette.success.main : theme.palette.error.main}
                    name="Return"
                  />
                </RechartsBarChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Equity Curve
            </Typography>
            
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={simulationResults.equity_curve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <RechartsTooltip />
                <Area 
                  type="monotone" 
                  dataKey="equity" 
                  stroke={theme.palette.primary.main}
                  fill={theme.palette.primary.light}
                />
              </AreaChart>
            </ResponsiveContainer>
            
            <Box mt={3}>
              <Typography variant="subtitle2" gutterBottom>
                Learning Insights
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <ul style={{ paddingLeft: theme.spacing(2) }}>
                  {simulationResults.learning_insights.map((insight, index) => (
                    <li key={index}>
                      <Typography variant="body2">{insight}</Typography>
                    </li>
                  ))}
                </ul>
              </Paper>
            </Box>
          </Grid>
        </Grid>
      </Box>
    );
  };

  return (
    <Paper elevation={0} variant="outlined">
      <Box p={2}>
        <Typography variant="h6" gutterBottom>Advanced Strategy Simulator</Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Simulate trading strategies with historical data and self-learning capabilities
        </Typography>
        
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
          sx={{ mb: 3 }}
        >
          <Tab label="Select Strategy" />
          <Tab label="Configure Simulation" />
          <Tab label="Results" />
        </Tabs>
        
        {activeTab === 0 && renderStrategyTemplates()}
        {activeTab === 1 && renderSimulationSettings()}
        {activeTab === 2 && renderSimulationResults()}
        
        <Divider sx={{ my: 3 }} />
        
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            {selectedStrategy && (
              <Typography variant="body2">
                Selected Strategy: <strong>{selectedStrategy.name}</strong>
              </Typography>
            )}
          </Box>
          
          <Box>
            {isSimulating ? (
              <>
                <Box display="flex" alignItems="center" mb={1}>
                  <CircularProgress size={24} sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    Simulating... {simulationProgress.toFixed(0)}%
                  </Typography>
                </Box>
                <Button
                  variant="outlined"
                  color="secondary"
                  startIcon={<Stop />}
                  onClick={handleStopSimulation}
                >
                  Stop Simulation
                </Button>
              </>
            ) : (
              <Button
                variant="contained"
                color="primary"
                startIcon={<PlayArrow />}
                onClick={handleStartSimulation}
                disabled={!selectedStrategy}
              >
                Run Simulation
              </Button>
            )}
          </Box>
        </Box>
      </Box>
    </Paper>
  );
};

export default StrategySimulator;