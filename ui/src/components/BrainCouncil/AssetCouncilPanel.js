import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardHeader, 
  CardContent, 
  Typography, 
  Grid, 
  Chip, 
  Box, 
  CircularProgress,
  Divider,
  Button,
  IconButton,
  Tooltip
} from '@mui/material';
import { 
  TrendingUp, 
  TrendingDown, 
  TrendingFlat,
  BarChart,
  BubbleChart,
  Timeline,
  Psychology,
  Insights,
  AutoGraph,
  Refresh
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip as RechartsTooltip } from 'recharts';
import { formatNumber, formatPercentage } from '../../utils/formatters';
import { getAssetColor } from '../../utils/assetUtils';
import api from '../../api';

const AssetCouncilPanel = ({ assetId, timeframe }) => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [councilData, setCouncilData] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const fetchCouncilData = async () => {
    setLoading(true);
    try {
      const response = await api.get(`/api/brain-council/asset/${assetId}?timeframe=${timeframe}`);
      setCouncilData(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching asset council data:', err);
      setError('Failed to load asset council data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCouncilData();
    // Set up polling interval
    const interval = setInterval(fetchCouncilData, 30000); // 30 seconds
    return () => clearInterval(interval);
  }, [assetId, timeframe]);

  const getDirectionIcon = (direction) => {
    switch (direction.toLowerCase()) {
      case 'buy':
      case 'bullish':
        return <TrendingUp style={{ color: theme.palette.success.main }} />;
      case 'sell':
      case 'bearish':
        return <TrendingDown style={{ color: theme.palette.error.main }} />;
      default:
        return <TrendingFlat style={{ color: theme.palette.text.secondary }} />;
    }
  };

  const getRegimeChip = (regime) => {
    let color;
    switch (regime) {
      case 'trending_bullish':
        color = theme.palette.success.main;
        break;
      case 'trending_bearish':
        color = theme.palette.error.main;
        break;
      case 'volatile':
        color = theme.palette.warning.main;
        break;
      case 'choppy':
        color = theme.palette.info.main;
        break;
      case 'ranging':
        color = theme.palette.secondary.main;
        break;
      default:
        color = theme.palette.text.secondary;
    }
    
    return (
      <Chip 
        label={regime.replace('_', ' ')} 
        size="small" 
        style={{ 
          backgroundColor: color,
          color: theme.palette.getContrastText(color),
          textTransform: 'capitalize'
        }} 
      />
    );
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return theme.palette.success.main;
    if (confidence >= 0.6) return theme.palette.success.light;
    if (confidence >= 0.4) return theme.palette.warning.main;
    return theme.palette.error.light;
  };

  const renderVotingBreakdown = () => {
    if (!councilData || !councilData.votes) return null;
    
    const voteData = Object.entries(councilData.votes).map(([source, vote]) => ({
      name: source,
      value: vote.confidence,
      direction: vote.direction
    }));
    
    const COLORS = {
      buy: theme.palette.success.main,
      sell: theme.palette.error.main,
      hold: theme.palette.text.secondary
    };
    
    return (
      <Box mt={2}>
        <Typography variant="subtitle2" gutterBottom>Voting Breakdown</Typography>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={voteData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
              nameKey="name"
            >
              {voteData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[entry.direction] || theme.palette.grey[500]} />
              ))}
            </Pie>
            <RechartsTooltip 
              formatter={(value, name) => [
                `Confidence: ${formatPercentage(value)}`,
                `Source: ${name}`
              ]}
            />
          </PieChart>
        </ResponsiveContainer>
        
        <Box mt={1}>
          {voteData.map((vote, index) => (
            <Box key={index} display="flex" alignItems="center" mb={0.5}>
              <Box 
                width={12} 
                height={12} 
                bgcolor={COLORS[vote.direction] || theme.palette.grey[500]} 
                mr={1} 
                borderRadius="50%" 
              />
              <Typography variant="body2" style={{ flexGrow: 1 }}>
                {vote.name}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {vote.direction} ({formatPercentage(vote.value)})
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    );
  };

  const renderModelContributions = () => {
    if (!councilData || !councilData.ml_models) return null;
    
    return (
      <Box mt={3}>
        <Typography variant="subtitle2" gutterBottom>ML Model Contributions</Typography>
        <Grid container spacing={1}>
          {councilData.ml_models.map((model, index) => (
            <Grid item xs={6} key={index}>
              <Box 
                p={1} 
                border={1} 
                borderColor="divider" 
                borderRadius={1}
                display="flex"
                alignItems="center"
              >
                <Box mr={1}>
                  {model.type === 'classification' && <BubbleChart fontSize="small" color="primary" />}
                  {model.type === 'regression' && <Timeline fontSize="small" color="secondary" />}
                  {model.type === 'deep_learning' && <Psychology fontSize="small" color="error" />}
                  {model.type === 'ensemble' && <Insights fontSize="small" color="success" />}
                </Box>
                <Box flexGrow={1}>
                  <Typography variant="body2" noWrap>{model.name}</Typography>
                  <Typography variant="caption" color="textSecondary">
                    {model.type} • {formatPercentage(model.accuracy)}
                  </Typography>
                </Box>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  if (loading && !councilData) {
    return (
      <Card>
        <CardHeader 
          title={`Asset Council: ${assetId}`}
          subheader={`Timeframe: ${timeframe}`}
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
          title={`Asset Council: ${assetId}`}
          subheader={`Timeframe: ${timeframe}`}
        />
        <CardContent>
          <Typography color="error">{error}</Typography>
          <Button 
            variant="outlined" 
            startIcon={<Refresh />} 
            onClick={fetchCouncilData}
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
            <Typography variant="h6" component="div">
              {assetId}
            </Typography>
            <Box ml={1}>
              {councilData && getDirectionIcon(councilData.direction)}
            </Box>
          </Box>
        }
        subheader={`Timeframe: ${timeframe}`}
        action={
          <IconButton onClick={fetchCouncilData} size="small">
            <Refresh fontSize="small" />
          </IconButton>
        }
      />
      <CardContent>
        {councilData && (
          <>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">Direction</Typography>
                <Box display="flex" alignItems="center">
                  {getDirectionIcon(councilData.direction)}
                  <Typography 
                    variant="body1" 
                    style={{ 
                      marginLeft: theme.spacing(0.5),
                      textTransform: 'capitalize'
                    }}
                  >
                    {councilData.direction}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">Confidence</Typography>
                <Box display="flex" alignItems="center">
                  <Box 
                    position="relative" 
                    display="inline-flex"
                    mr={1}
                  >
                    <CircularProgress 
                      variant="determinate" 
                      value={councilData.confidence * 100} 
                      size={24}
                      style={{ color: getConfidenceColor(councilData.confidence) }}
                    />
                    <Box
                      top={0}
                      left={0}
                      bottom={0}
                      right={0}
                      position="absolute"
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                    >
                      <Typography variant="caption" component="div" color="textSecondary">
                        {Math.round(councilData.confidence * 100)}%
                      </Typography>
                    </Box>
                  </Box>
                  <Typography variant="body1">
                    {formatPercentage(councilData.confidence)}
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            <Box mt={2}>
              <Typography variant="body2" color="textSecondary">Market Regime</Typography>
              <Box mt={0.5}>
                {getRegimeChip(councilData.regime)}
              </Box>
            </Box>

            <Divider style={{ margin: `${theme.spacing(2)} 0` }} />
            
            {expanded ? (
              <>
                {renderVotingBreakdown()}
                {renderModelContributions()}
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

export default AssetCouncilPanel;