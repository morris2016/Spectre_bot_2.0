import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  CircularProgress, 
  FormControl, 
  Select, 
  MenuItem, 
  InputLabel,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Chip,
  Divider
} from '@mui/material';
import { 
  Refresh, 
  ZoomIn, 
  ZoomOut, 
  Timeline, 
  ShowChart, 
  Visibility, 
  VisibilityOff,
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Flag
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { createChart, CrosshairMode } from 'lightweight-charts';
import api from '../../api';

const PredictiveChart = ({ 
  asset, 
  platform = 'binance', 
  timeframe = '1h',
  height = 500,
  showPredictions = true,
  showPatterns = true,
  onPredictionUpdate = () => {}
}) => {
  const theme = useTheme();
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeries = useRef(null);
  const volumeSeries = useRef(null);
  const predictionSeries = useRef(null);
  const patternMarkers = useRef([]);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [patterns, setPatterns] = useState([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [isPredictionsVisible, setIsPredictionsVisible] = useState(showPredictions);
  const [isPatternsVisible, setIsPatternsVisible] = useState(showPatterns);
  const [latestPrediction, setLatestPrediction] = useState(null);
  const [chartType, setChartType] = useState('candles');

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        backgroundColor: theme.palette.background.paper,
        textColor: theme.palette.text.primary,
      },
      grid: {
        vertLines: {
          color: theme.palette.divider,
        },
        horzLines: {
          color: theme.palette.divider,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: theme.palette.divider,
      },
      timeScale: {
        borderColor: theme.palette.divider,
        timeVisible: true,
      },
    });

    // Create series
    if (chartType === 'candles') {
      candleSeries.current = chart.addCandlestickSeries({
        upColor: theme.palette.success.main,
        downColor: theme.palette.error.main,
        borderDownColor: theme.palette.error.main,
        borderUpColor: theme.palette.success.main,
        wickDownColor: theme.palette.error.main,
        wickUpColor: theme.palette.success.main,
      });
    } else {
      candleSeries.current = chart.addLineSeries({
        color: theme.palette.primary.main,
        lineWidth: 2,
      });
    }

    // Add volume series
    volumeSeries.current = chart.addHistogramSeries({
      color: theme.palette.primary.light,
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Add prediction series
    predictionSeries.current = chart.addLineSeries({
      color: theme.palette.secondary.main,
      lineWidth: 2,
      lineStyle: 1, // Dashed
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // Store chart reference
    chartRef.current = chart;

    // Handle resize
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({ 
          width: chartContainerRef.current.clientWidth 
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Fetch initial data
    fetchData();

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        candleSeries.current = null;
        volumeSeries.current = null;
        predictionSeries.current = null;
      }
    };
  }, [asset, platform, chartType]);

  // Update data when timeframe changes
  useEffect(() => {
    fetchData();
  }, [selectedTimeframe]);

  // Update predictions visibility
  useEffect(() => {
    if (predictionSeries.current) {
      predictionSeries.current.applyOptions({
        visible: isPredictionsVisible
      });
    }
  }, [isPredictionsVisible]);

  // Update patterns visibility
  useEffect(() => {
    if (chartRef.current) {
      if (isPatternsVisible) {
        renderPatternMarkers();
      } else {
        clearPatternMarkers();
      }
    }
  }, [isPatternsVisible, patterns]);

  // Fetch chart data
  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch OHLC data
      const ohlcResponse = await api.get(`/api/market/ohlc/${platform}/${asset}/${selectedTimeframe}?limit=500`);
      
      if (ohlcResponse.data && ohlcResponse.data.data) {
        const formattedData = ohlcResponse.data.data.map(candle => ({
          time: candle.timestamp / 1000,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          volume: candle.volume
        }));

        setData(formattedData);

        // Update chart series
        if (candleSeries.current) {
          if (chartType === 'candles') {
            candleSeries.current.setData(formattedData);
          } else {
            candleSeries.current.setData(
              formattedData.map(item => ({
                time: item.time,
                value: item.close
              }))
            );
          }
        }

        if (volumeSeries.current) {
          volumeSeries.current.setData(
            formattedData.map(item => ({
              time: item.time,
              value: item.volume,
              color: item.close >= item.open 
                ? theme.palette.success.light 
                : theme.palette.error.light
            }))
          );
        }

        // Fetch predictions
        fetchPredictions();
        
        // Fetch patterns
        fetchPatterns();
      }
    } catch (err) {
      console.error('Error fetching chart data:', err);
      setError('Failed to load chart data');
    } finally {
      setLoading(false);
    }
  };

  // Fetch price predictions
  const fetchPredictions = async () => {
    try {
      const response = await api.get(`/api/brain-council/predictions/${platform}/${asset}/${selectedTimeframe}`);
      
      if (response.data && response.data.predictions) {
        const predictionData = response.data.predictions;
        setPredictions(predictionData);
        
        // Set latest prediction for parent component
        if (predictionData.length > 0) {
          const latest = predictionData[predictionData.length - 1];
          setLatestPrediction(latest);
          onPredictionUpdate(latest);
        }

        // Format for chart
        if (predictionSeries.current) {
          const lastCandle = data[data.length - 1];
          
          if (lastCandle) {
            const chartData = [
              { time: lastCandle.time, value: lastCandle.close }
            ];
            
            // Add future predictions
            predictionData.forEach(pred => {
              chartData.push({
                time: pred.timestamp,
                value: pred.price
              });
            });
            
            predictionSeries.current.setData(chartData);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching predictions:', err);
    }
  };

  // Fetch chart patterns
  const fetchPatterns = async () => {
    try {
      const response = await api.get(`/api/brain-council/patterns/${platform}/${asset}/${selectedTimeframe}`);
      
      if (response.data && response.data.patterns) {
        setPatterns(response.data.patterns);
        
        if (isPatternsVisible) {
          renderPatternMarkers();
        }
      }
    } catch (err) {
      console.error('Error fetching patterns:', err);
    }
  };

  // Render pattern markers on chart
  const renderPatternMarkers = () => {
    // Clear existing markers
    clearPatternMarkers();
    
    if (!chartRef.current) return;
    
    // Add new markers
    patterns.forEach(pattern => {
      const marker = {
        time: pattern.time,
        position: pattern.type.includes('bullish') ? 'belowBar' : 'aboveBar',
        color: pattern.type.includes('bullish') ? theme.palette.success.main : theme.palette.error.main,
        shape: pattern.type.includes('bullish') ? 'arrowUp' : 'arrowDown',
        text: pattern.name
      };
      
      patternMarkers.current.push(marker);
      candleSeries.current.setMarker(marker);
    });
  };

  // Clear pattern markers
  const clearPatternMarkers = () => {
    if (candleSeries.current) {
      patternMarkers.current.forEach(marker => {
        candleSeries.current.removeMarker(marker);
      });
      patternMarkers.current = [];
    }
  };

  // Handle timeframe change
  const handleTimeframeChange = (event) => {
    setSelectedTimeframe(event.target.value);
  };

  // Handle chart type change
  const handleChartTypeChange = (type) => {
    setChartType(type);
    
    // Re-initialize chart with new type
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      candleSeries.current = null;
      volumeSeries.current = null;
      predictionSeries.current = null;
    }
  };

  // Handle zoom in/out
  const handleZoom = (direction) => {
    if (!chartRef.current) return;
    
    const timeScale = chartRef.current.timeScale();
    if (direction === 'in') {
      timeScale.zoomIn();
    } else {
      timeScale.zoomOut();
    }
  };

  // Render prediction info
  const renderPredictionInfo = () => {
    if (!latestPrediction) return null;
    
    const direction = latestPrediction.direction;
    const confidence = latestPrediction.confidence;
    const priceDiff = latestPrediction.price_diff_percent;
    
    let directionIcon;
    let directionColor;
    
    if (direction === 'bullish') {
      directionIcon = <TrendingUp />;
      directionColor = theme.palette.success.main;
    } else if (direction === 'bearish') {
      directionIcon = <TrendingDown />;
      directionColor = theme.palette.error.main;
    } else {
      directionIcon = <TrendingFlat />;
      directionColor = theme.palette.text.secondary;
    }
    
    return (
      <Box 
        position="absolute" 
        top={16} 
        right={16} 
        bgcolor="rgba(0,0,0,0.7)" 
        p={2} 
        borderRadius={1}
        zIndex={10}
      >
        <Typography variant="subtitle2" color="white">Prediction</Typography>
        <Box display="flex" alignItems="center" mt={1}>
          <Box color={directionColor} mr={1}>
            {directionIcon}
          </Box>
          <Typography variant="body2" color="white" sx={{ textTransform: 'capitalize' }}>
            {direction} ({confidence.toFixed(2) * 100}%)
          </Typography>
        </Box>
        <Typography variant="body2" color="white" mt={0.5}>
          Target: {priceDiff > 0 ? '+' : ''}{priceDiff.toFixed(2)}%
        </Typography>
      </Box>
    );
  };

  if (loading && !data.length) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        height={height} 
        border={1} 
        borderColor="divider" 
        borderRadius={1}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        height={height} 
        border={1} 
        borderColor="divider" 
        borderRadius={1}
      >
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Paper elevation={0} variant="outlined">
      <Box p={2} display="flex" justifyContent="space-between" alignItems="center">
        <Box>
          <Typography variant="h6">{asset}</Typography>
          <Typography variant="body2" color="textSecondary">{platform}</Typography>
        </Box>
        
        <Box display="flex" alignItems="center">
          <FormControl size="small" variant="outlined" sx={{ minWidth: 120, mr: 2 }}>
            <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
            <Select
              labelId="timeframe-select-label"
              value={selectedTimeframe}
              onChange={handleTimeframeChange}
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
          
          <Box mr={2}>
            <Tooltip title="Candlestick Chart">
              <IconButton 
                color={chartType === 'candles' ? 'primary' : 'default'} 
                onClick={() => handleChartTypeChange('candles')}
              >
                <ShowChart />
              </IconButton>
            </Tooltip>
            <Tooltip title="Line Chart">
              <IconButton 
                color={chartType === 'line' ? 'primary' : 'default'} 
                onClick={() => handleChartTypeChange('line')}
              >
                <Timeline />
              </IconButton>
            </Tooltip>
          </Box>
          
          <Box mr={2}>
            <Tooltip title="Zoom In">
              <IconButton onClick={() => handleZoom('in')}>
                <ZoomIn />
              </IconButton>
            </Tooltip>
            <Tooltip title="Zoom Out">
              <IconButton onClick={() => handleZoom('out')}>
                <ZoomOut />
              </IconButton>
            </Tooltip>
          </Box>
          
          <Box mr={2}>
            <FormControlLabel
              control={
                <Switch
                  checked={isPredictionsVisible}
                  onChange={(e) => setIsPredictionsVisible(e.target.checked)}
                  color="primary"
                />
              }
              label="Predictions"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={isPatternsVisible}
                  onChange={(e) => setIsPatternsVisible(e.target.checked)}
                  color="primary"
                />
              }
              label="Patterns"
            />
          </Box>
          
          <Tooltip title="Refresh">
            <IconButton onClick={fetchData}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      <Divider />
      
      <Box position="relative">
        {isPredictionsVisible && renderPredictionInfo()}
        <div ref={chartContainerRef} />
      </Box>
      
      {isPatternsVisible && patterns.length > 0 && (
        <Box p={2} display="flex" flexWrap="wrap" gap={1}>
          {patterns.map((pattern, index) => (
            <Chip
              key={index}
              icon={pattern.type.includes('bullish') ? <TrendingUp /> : <TrendingDown />}
              label={pattern.name}
              color={pattern.type.includes('bullish') ? 'success' : 'error'}
              size="small"
            />
          ))}
        </Box>
      )}
    </Paper>
  );
};

export default PredictiveChart;