import React, { useEffect, useRef, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import PropTypes from 'prop-types';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { 
  addPatternMarker,
  updateTradeSignals,
  setTimeframe,
  setChartReady
} from '../../store/chartSlice';
import { selectCurrentAsset } from '../../store/assetSlice';
import PatternOverlay from './PatternOverlay';
import SignalIndicator from './SignalIndicator';
import TimeframeSelector from './TimeframeSelector';
import WebSocket from '../../services/websocket';
import PatternRecognitionService from '../../services/patternRecognition';
import './TradingView.scss';

/**
 * Advanced trading chart component with pattern recognition, multi-timeframe analysis,
 * and auto-trading signal indicators.
 * 
 * This component integrates with the Brain Council for real-time signals and
 * Intelligence System for pattern recognition directly on charts.
 */
const TradingView = ({ 
  containerId = 'tv_chart_container',
  platform = 'binance',
  onPatternDetected = () => {},
  onSignalGenerated = () => {},
  height = '100%',
  width = '100%',
  showToolbar = true,
  autoScale = true,
  darkMode = true
}) => {
  const chartContainerRef = useRef();
  const chartInstanceRef = useRef(null);
  const candleSeries = useRef(null);
  const volumeSeries = useRef(null);
  const dispatch = useDispatch();

  // Redux state selectors
  const currentAsset = useSelector(selectCurrentAsset);
  const timeframe = useSelector(state => state.chart.timeframe);
  const signals = useSelector(state => state.chart.signals);
  const patterns = useSelector(state => state.chart.patterns);
  const brainSuggestions = useSelector(state => state.brainCouncil.suggestions);
  const orderFlowData = useSelector(state => state.market.orderFlow);
  const tradingSettings = useSelector(state => state.settings.trading);
  
  // Local state
  const [chartReady, setChartReadyState] = useState(false);
  const [visibleRange, setVisibleRange] = useState({ from: null, to: null });
  const [hoverData, setHoverData] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [markers, setMarkers] = useState([]);
  const [drawing, setDrawing] = useState(null);
  
  // Color scheme based on dark mode
  const colors = darkMode ? {
    background: '#1a1a1a',
    text: '#d1d4dc',
    grid: '#2a2a2a',
    upCandle: '#26a69a',
    downCandle: '#ef5350',
    volume: {
      up: 'rgba(38, 166, 154, 0.3)',
      down: 'rgba(239, 83, 80, 0.3)'
    },
    crosshair: '#758696',
    watermark: 'rgba(255, 255, 255, 0.1)'
  } : {
    background: '#ffffff',
    text: '#132235',
    grid: '#f0f3fa',
    upCandle: '#26a69a',
    downCandle: '#ef5350',
    volume: {
      up: 'rgba(38, 166, 154, 0.3)',
      down: 'rgba(239, 83, 80, 0.3)'
    },
    crosshair: '#9db2bd',
    watermark: 'rgba(0, 0, 0, 0.1)'
  };

  /**
   * Initialize the chart instance
   */
  const initializeChart = () => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
    }

    chartInstanceRef.current = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        backgroundColor: colors.background,
        textColor: colors.text,
        fontFamily: 'Inter, sans-serif'
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid }
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: colors.crosshair,
          width: 1,
          style: 0,
          labelBackgroundColor: colors.crosshair
        },
        horzLine: {
          color: colors.crosshair,
          width: 1,
          style: 0,
          labelBackgroundColor: colors.crosshair
        }
      },
      rightPriceScale: {
        borderColor: colors.grid,
        autoScale: autoScale,
      },
      timeScale: {
        borderColor: colors.grid,
        timeVisible: true,
        secondsVisible: false,
      },
      watermark: {
        visible: true,
        text: 'QuantumSpectre Elite',
        color: colors.watermark,
        fontSize: 36,
        horzAlign: 'center',
        vertAlign: 'center',
      }
    });

    // Create the main candle series
    candleSeries.current = chartInstanceRef.current.addCandlestickSeries({
      upColor: colors.upCandle,
      downColor: colors.downCandle,
      borderVisible: false,
      wickUpColor: colors.upCandle,
      wickDownColor: colors.downCandle
    });

    // Create volume series as histogram below main chart
    volumeSeries.current = chartInstanceRef.current.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Configure the volume scale to be overlaid
    chartInstanceRef.current.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
      autoScale: true,
      visible: false
    });

    // Event listeners
    chartInstanceRef.current.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) {
        setVisibleRange({
          from: range.from,
          to: range.to
        });
      }
    });

    chartInstanceRef.current.subscribeCrosshairMove((param) => {
      if (param && param.time) {
        const data = param.seriesData.get(candleSeries.current);
        if (data) {
          setHoverData({
            time: param.time,
            ...data
          });
        }
      } else {
        setHoverData(null);
      }
    });

    // Setup resize handler
    const handleResize = () => {
      if (chartInstanceRef.current && chartContainerRef.current) {
        chartInstanceRef.current.resize(
          chartContainerRef.current.clientWidth,
          chartContainerRef.current.clientHeight
        );
      }
    };
    
    window.addEventListener('resize', handleResize);
    
    // Set chart ready
    setChartReadyState(true);
    dispatch(setChartReady(true));
    
    return () => window.removeEventListener('resize', handleResize);
  };

  /**
   * Load data for the current asset and timeframe
   */
  const loadChartData = async () => {
    if (!currentAsset || !timeframe) return;
    
    try {
      // Show loading state
      setChartData([]);
      
      // Fetch historical data
      const response = await fetch(`/api/market-data/${platform}/${currentAsset.symbol}?timeframe=${timeframe}`);
      const data = await response.json();
      
      if (data && data.candles && candleSeries.current) {
        // Process candle data to include timestamps
        const processedCandles = data.candles.map(candle => ({
          time: candle.timestamp / 1000,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close
        }));
        
        // Process volume data
        const processedVolumes = data.candles.map(candle => ({
          time: candle.timestamp / 1000,
          value: candle.volume,
          color: candle.close >= candle.open ? colors.volume.up : colors.volume.down
        }));
        
        // Store the processed data
        setChartData(processedCandles);
        
        // Set the chart data
        candleSeries.current.setData(processedCandles);
        volumeSeries.current.setData(processedVolumes);
        
        // Apply the patterns and signals from the store
        applyPatterns();
        applySignals();
      }
    } catch (error) {
      console.error('Error loading chart data:', error);
      // Show error toast
    }
  };
  
  /**
   * Connect to websocket for real-time updates
   */
  const connectRealTimeData = () => {
    if (!currentAsset || !timeframe) return;
    
    // Disconnect any existing connection
    WebSocket.disconnect(`market_${platform}_${currentAsset.symbol}_${timeframe}`);
    
    // Connect to the websocket feed
    WebSocket.connect(`market_${platform}_${currentAsset.symbol}_${timeframe}`, (data) => {
      if (data && data.type === 'candle' && candleSeries.current) {
        const candle = {
          time: data.timestamp / 1000,
          open: data.open,
          high: data.high,
          low: data.low,
          close: data.close
        };
        
        const volume = {
          time: data.timestamp / 1000,
          value: data.volume,
          color: data.close >= data.open ? colors.volume.up : colors.volume.down
        };
        
        // Update the series
        candleSeries.current.update(candle);
        volumeSeries.current.update(volume);
        
        // Process for pattern recognition
        processForPatterns(candle);
      } else if (data && data.type === 'trade') {
        // Process individual trades for order flow analysis
        processOrderFlow(data);
      } else if (data && data.type === 'brain_signal') {
        // Process brain signals
        processBrainSignal(data);
      }
    });
  };

  /**
   * Process candles for pattern recognition
   */
  const processForPatterns = (candle) => {
    // Get last N candles for pattern recognition
    const lastCandles = [...chartData.slice(-50), candle];
    
    // Use the PatternRecognitionService to detect patterns
    const detectedPatterns = PatternRecognitionService.detectPatterns(lastCandles);
    
    if (detectedPatterns && detectedPatterns.length > 0) {
      // Add each detected pattern to the chart
      detectedPatterns.forEach(pattern => {
        const marker = {
          time: candle.time,
          position: pattern.position,
          color: pattern.bullish ? '#26a69a' : '#ef5350',
          shape: pattern.bullish ? 'arrowUp' : 'arrowDown',
          text: pattern.name,
          size: 1
        };
        
        // Add marker to chart
        candleSeries.current.setMarkers([...markers, marker]);
        setMarkers(prevMarkers => [...prevMarkers, marker]);
        
        // Dispatch to store
        dispatch(addPatternMarker({
          id: `${pattern.name}_${candle.time}`,
          patternType: pattern.name,
          position: pattern.position,
          time: candle.time,
          strength: pattern.strength,
          bullish: pattern.bullish
        }));
        
        // Notify callback
        onPatternDetected(pattern, candle);
      });
    }
  };

  /**
   * Process order flow data
   */
  const processOrderFlow = (trade) => {
    // Implementation for order flow analysis
    // This would feed into advanced heatmaps and pressure indicators
  };

  /**
   * Process brain signals from the Brain Council
   */
  const processBrainSignal = (signal) => {
    // Add the signal to the chart
    const signalMarker = {
      time: signal.time,
      position: signal.action === 'buy' ? 'belowBar' : 'aboveBar',
      color: signal.action === 'buy' ? '#26a69a' : '#ef5350',
      shape: signal.action === 'buy' ? 'arrowUp' : 'arrowDown',
      text: `${signal.action.toUpperCase()} - ${signal.confidence.toFixed(2)}%`,
      size: signal.confidence / 20  // Scale marker size based on confidence
    };
    
    // Add to chart
    candleSeries.current.setMarkers([...markers, signalMarker]);
    setMarkers(prevMarkers => [...prevMarkers, signalMarker]);
    
    // Dispatch to store
    dispatch(updateTradeSignals({
      id: `signal_${signal.time}`,
      action: signal.action,
      confidence: signal.confidence,
      time: signal.time,
      source: 'brain_council',
      reasons: signal.reasons || []
    }));
    
    // Notify callback
    onSignalGenerated(signal);
  };

  /**
   * Apply patterns from the store to the chart
   */
  const applyPatterns = () => {
    if (!candleSeries.current || !patterns || patterns.length === 0) return;
    
    const patternMarkers = patterns.map(pattern => ({
      time: pattern.time,
      position: pattern.position,
      color: pattern.bullish ? '#26a69a' : '#ef5350',
      shape: pattern.bullish ? 'arrowUp' : 'arrowDown',
      text: pattern.patternType,
      size: pattern.strength / 10
    }));
    
    candleSeries.current.setMarkers(patternMarkers);
    setMarkers(patternMarkers);
  };

  /**
   * Apply signals from the store to the chart
   */
  const applySignals = () => {
    if (!candleSeries.current || !signals || signals.length === 0) return;
    
    const signalMarkers = signals.map(signal => ({
      time: signal.time,
      position: signal.action === 'buy' ? 'belowBar' : 'aboveBar',
      color: signal.action === 'buy' ? '#26a69a' : '#ef5350',
      shape: signal.action === 'buy' ? 'arrowUp' : 'arrowDown',
      text: `${signal.action.toUpperCase()} - ${signal.confidence.toFixed(2)}%`,
      size: signal.confidence / 20
    }));
    
    // Add the signals to the existing markers
    const allMarkers = [...markers.filter(m => !m.text.includes('BUY') && !m.text.includes('SELL')), ...signalMarkers];
    candleSeries.current.setMarkers(allMarkers);
    setMarkers(allMarkers);
  };

  /**
   * Handle timeframe change
   */
  const handleTimeframeChange = (newTimeframe) => {
    dispatch(setTimeframe(newTimeframe));
  };

  /**
   * Enable drawing mode for chart annotations
   */
  const enableDrawingMode = (type) => {
    if (!chartInstanceRef.current) return;
    
    // Clear any active drawing
    if (drawing) {
      chartInstanceRef.current.removeDrawingTool();
    }
    
    // Initialize the new drawing tool
    const drawingTool = chartInstanceRef.current.createDrawingTool();
    drawingTool.setMode(type);
    setDrawing(type);
  };

  /**
   * Handle drawing completion
   */
  const handleDrawingComplete = (drawingData) => {
    // Save the drawing to local storage or backend
    console.log('Drawing completed:', drawingData);
    setDrawing(null);
  };

  /**
   * Initialize chart when component mounts
   */
  useEffect(() => {
    if (chartContainerRef.current) {
      const chartInit = initializeChart();
      return () => {
        // Clean up
        WebSocket.disconnectAll();
        if (chartInstanceRef.current) {
          chartInstanceRef.current.remove();
        }
        if (typeof chartInit === 'function') {
          chartInit();
        }
      };
    }
  }, [darkMode]); // Reinitialize chart when dark mode changes

  /**
   * Load data when asset or timeframe changes
   */
  useEffect(() => {
    if (chartReady && currentAsset && timeframe) {
      loadChartData();
      connectRealTimeData();
    }
  }, [chartReady, currentAsset, timeframe, platform]);

  /**
   * Apply patterns and signals when they change
   */
  useEffect(() => {
    if (chartReady && candleSeries.current) {
      applyPatterns();
    }
  }, [chartReady, patterns]);

  useEffect(() => {
    if (chartReady && candleSeries.current) {
      applySignals();
    }
  }, [chartReady, signals]);

  /**
   * Apply brain suggestions when they change
   */
  useEffect(() => {
    if (chartReady && candleSeries.current && brainSuggestions && brainSuggestions.length > 0) {
      // Process brain suggestions for visualization
      const latestSuggestion = brainSuggestions[0];
      processBrainSignal({
        time: Date.now() / 1000,
        action: latestSuggestion.action,
        confidence: latestSuggestion.confidence,
        reasons: latestSuggestion.reasons
      });
    }
  }, [chartReady, brainSuggestions]);

  /**
   * Apply order flow data when it changes
   */
  useEffect(() => {
    if (chartReady && candleSeries.current && orderFlowData) {
      // Visualize order flow data (heatmap, pressure, etc.)
    }
  }, [chartReady, orderFlowData]);

  return (
    
{showToolbar && ( enableDrawingMode('line')} title="Line Tool" > enableDrawingMode('horizontal')} title="Horizontal Line" > enableDrawingMode('fibonacci')} title="Fibonacci Retracement" > enableDrawingMode('rectangle')} title="Rectangle" > {drawing && ( setDrawing(null)} title="Cancel Drawing" > )} )} {hoverData && ( O: {hoverData.open.toFixed(8)} H: {hoverData.high.toFixed(8)} L: {hoverData.low.toFixed(8)} C: {hoverData.close.toFixed(8)} )} {/* Signal indicators for latest brain recommendations */} {/* Pattern overlay component */} p.time >= visibleRange.from && p.time <= visibleRange.to )} visible={tradingSettings.showPatternOverlay} /> ); }; TradingView.propTypes = { containerId: PropTypes.string, platform: PropTypes.oneOf(['binance', 'deriv']), onPatternDetected: PropTypes.func, onSignalGenerated: PropTypes.func, height: PropTypes.string, width: PropTypes.string, showToolbar: PropTypes.bool, autoScale: PropTypes.bool, darkMode: PropTypes.bool }; export default TradingView;