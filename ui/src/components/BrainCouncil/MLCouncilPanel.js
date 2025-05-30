import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardHeader, 
  CardContent, 
  Typography, 
  Grid, 
  Box, 
  CircularProgress,
  Divider,
  Button,
  IconButton,
  Tooltip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress
} from '@mui/material';
import { 
  Refresh,
  Psychology,
  BubbleChart,
  Timeline,
  Insights,
  AutoGraph,
  BarChart,
  ShowChart,
  Memory
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { 
  ResponsiveContainer, 
  BarChart as RechartsBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip,
  Legend
} from 'recharts';
import { formatNumber, formatPercentage } from '../../utils/formatters';
import api from '../../api';

const MLCouncilPanel = () => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [mlCouncilData, setMlCouncilData] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const fetchMlCouncilData = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/brain-council/ml-council');
      setMlCouncilData(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching ML council data:', err);
      setError('Failed to load ML council data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMlCouncilData();
    // Set up polling interval
    const interval = setInterval(fetchMlCouncilData, 60000); // 60 seconds
    return () => clearInterval(interval);
  }, []);

  const getModelTypeIcon = (type) => {
    switch (type.toLowerCase()) {
      case 'classification':
        return <BubbleChart fontSize="small" color="primary" />;
      case 'regression':
        return <Timeline fontSize="small" color="secondary" />;
      case 'time_series':
        return <ShowChart fontSize="small" color="info" />;
      case 'deep_learning':
        return <Psychology fontSize="small" color="error" />;
      case 'reinforcement_learning':
        return <AutoGraph fontSize="small" color="warning" />;
      case 'ensemble':
        return <Insights fontSize="small" color="success" />;
      default:
        return <BarChart fontSize="small" color="action" />;
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 0.8) return theme.palette.success.main;
    if (accuracy >= 0.6) return theme.palette.success.light;
    if (accuracy >= 0.4) return theme.palette.warning.main;
    return theme.palette.error.light;
  };

  const renderModelTable = () => {
    if (!mlCouncilData || !mlCouncilData.models) return null;
    
    return (
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Model</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Accuracy</TableCell>
              <TableCell>Assets</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {mlCouncilData.models.map((model, index) => (
              <TableRow key={index}>
                <TableCell>
                  <Box display="flex" alignItems="center">
                    {getModelTypeIcon(model.type)}
                    <Typography variant="body2" style={{ marginLeft: theme.spacing(1) }}>
                      {model.name}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" style={{ textTransform: 'capitalize' }}>
                    {model.type.replace('_', ' ')}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Box display="flex" alignItems="center">
                    <LinearProgress
                      variant="determinate"
                      value={model.accuracy * 100}
                      style={{ 
                        width: 60, 
                        marginRight: theme.spacing(1),
                        backgroundColor: theme.palette.grey[300],
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: getAccuracyColor(model.accuracy)
                        }
                      }}
                    />
                    <Typography variant="body2">
                      {formatPercentage(model.accuracy)}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {model.asset_count || 'All'}
                  </Typography>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  const renderPerformanceChart = () => {
    if (!mlCouncilData || !mlCouncilData.performance) return null;
    
    const chartData = Object.entries(mlCouncilData.performance).map(([modelType, metrics]) => ({
      name: modelType.replace('_', ' '),
      accuracy: metrics.accuracy * 100,
      precision: metrics.precision * 100,
      recall: metrics.recall * 100,
      f1: metrics.f1_score * 100
    }));
    
    return (
      <Box mt={3}>
        <Typography variant="subtitle2" gutterBottom>Model Type Performance</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <RechartsBarChart
            data={chartData}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }} />
            <RechartsTooltip />
            <Legend />
            <Bar dataKey="accuracy" name="Accuracy" fill={theme.palette.primary.main} />
            <Bar dataKey="precision" name="Precision" fill={theme.palette.secondary.main} />
            <Bar dataKey="recall" name="Recall" fill={theme.palette.success.main} />
            <Bar dataKey="f1" name="F1 Score" fill={theme.palette.info.main} />
          </RechartsBarChart>
        </ResponsiveContainer>
      </Box>
    );
  };

  const renderAssetSpecificModels = () => {
    if (!mlCouncilData || !mlCouncilData.asset_models) return null;
    
    return (
      <Box mt={3}>
        <Typography variant="subtitle2" gutterBottom>Asset-Specific Models</Typography>
        <Grid container spacing={2}>
          {Object.entries(mlCouncilData.asset_models).map(([asset, models], index) => (
            <Grid item xs={12} md={6} key={index}>
              <Paper variant="outlined" style={{ padding: theme.spacing(1) }}>
                <Typography variant="body2" fontWeight="bold">{asset}</Typography>
                <Box mt={1}>
                  {models.map((model, idx) => (
                    <Box 
                      key={idx} 
                      display="flex" 
                      alignItems="center" 
                      mb={0.5}
                      p={0.5}
                      borderRadius={1}
                      bgcolor={theme.palette.action.hover}
                    >
                      {getModelTypeIcon(model.type)}
                      <Typography variant="caption" style={{ marginLeft: theme.spacing(0.5), flexGrow: 1 }}>
                        {model.name}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {formatPercentage(model.accuracy)}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  if (loading && !mlCouncilData) {
    return (
      <Card>
        <CardHeader 
          title="ML Council"
          subheader="Machine Learning Models Integration"
        />
        <CardContent>
          <Box display="flex" justifyContent="center" p={3}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader 
          title="ML Council"
          subheader="Machine Learning Models Integration"
        />
        <CardContent>
          <Typography color="error">{error}</Typography>
          <Button 
            variant="outlined" 
            startIcon={<Refresh />} 
            onClick={fetchMlCouncilData}
            style={{ marginTop: theme.spacing(2) }}
          >
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader 
        title={
          <Box display="flex" alignItems="center">
            <Memory color="primary" style={{ marginRight: theme.spacing(1) }} />
            <Typography variant="h6" component="div">
              ML Council
            </Typography>
          </Box>
        }
        subheader={`${mlCouncilData?.models?.length || 0} active models`}
        action={
          <IconButton onClick={fetchMlCouncilData} size="small">
            <Refresh fontSize="small" />
          </IconButton>
        }
      />
      <CardContent>
        {mlCouncilData && (
          <>
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Total Models</Typography>
                <Typography variant="h6">{mlCouncilData.models?.length || 0}</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Avg. Accuracy</Typography>
                <Typography variant="h6">{formatPercentage(mlCouncilData.avg_accuracy || 0)}</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Asset Coverage</Typography>
                <Typography variant="h6">{mlCouncilData.asset_coverage || 0}</Typography>
              </Grid>
            </Grid>

            <Divider style={{ margin: `${theme.spacing(2)} 0` }} />
            
            {expanded ? (
              <>
                {renderModelTable()}
                {renderPerformanceChart()}
                {renderAssetSpecificModels()}
                <Box mt={2} display="flex" justifyContent="center">
                  <Button 
                    size="small" 
                    onClick={() => setExpanded(false)}
                  >
                    Show Less
                  </Button>
                </Box>
              </>
            ) : (
              <Box mt={1} display="flex" justifyContent="center">
                <Button 
                  size="small" 
                  onClick={() => setExpanded(true)}
                >
                  Show Details
                </Button>
              </Box>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default MLCouncilPanel;