import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import * as d3 from 'd3';
import { ResizableBox } from 'react-resizable';
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

import { 
    TradingViewChart, 
    OrderBook, 
    TradeHistory, 
    AssetList, 
    PortfolioSummary,
    TradingControls,
    StrategySelector,
    PatternRecognition,
    VolumeProfile,
    MarketRegimeIndicator,
    SignalPanel,
    PnLChart,
    AlertPanel,
    NewsPanel,
    BrainActivity,
    VoiceAdvisor,
    SystemStatus,
    PlatformSelector,
    CorrelationMatrix,
    RiskManager
} from '../components';

import { 
    fetchMarketData, 
    updateActiveAsset, 
    togglePlatform, 
    subscribeToOrderBook, 
    subscribeToTrades,
    startStrategySimulation,
    requestVoiceAdvice,
    updateTimeframe
} from '../../actions';

import { 
    formatCurrency, 
    calculateDrawdown, 
    formatPercentage, 
    getThemeColors 
} from '../../utils';

import './Dashboard.scss';

const ResponsiveGridLayout = WidthProvider(Responsive);

// Default layout configurations for different screen sizes
const layouts = {
    lg: [
        { i: 'chart', x: 0, y: 0, w: 8, h: 16 },
        { i: 'orderbook', x: 8, y: 0, w: 4, h: 8 },
        { i: 'trades', x: 8, y: 8, w: 4, h: 8 },
        { i: 'assets', x: 0, y: 16, w: 4, h: 8 },
        { i: 'portfolio', x: 4, y: 16, w: 4, h: 8 },
        { i: 'controls', x: 8, y: 16, w: 4, h: 8 },
        { i: 'strategies', x: 0, y: 24, w: 4, h: 8 },
        { i: 'patterns', x: 4, y: 24, w: 4, h: 8 },
        { i: 'volume', x: 8, y: 24, w: 4, h: 8 },
        { i: 'regimes', x: 0, y: 32, w: 4, h: 8 },
        { i: 'signals', x: 4, y: 32, w: 4, h: 8 },
        { i: 'pnl', x: 8, y: 32, w: 4, h: 8 },
        { i: 'alerts', x: 0, y: 40, w: 4, h: 8 },
        { i: 'news', x: 4, y: 40, w: 4, h: 8 },
        { i: 'brain', x: 8, y: 40, w: 4, h: 8 },
        { i: 'voice', x: 0, y: 48, w: 12, h: 4 },
        { i: 'platform', x: 0, y: 52, w: 6, h: 4 },
        { i: 'status', x: 6, y: 52, w: 6, h: 4 },
        { i: 'correlation', x: 0, y: 56, w: 6, h: 8 },
        { i: 'risk', x: 6, y: 56, w: 6, h: 8 }
    ],
    md: [
        { i: 'chart', x: 0, y: 0, w: 8, h: 14 },
        { i: 'orderbook', x: 8, y: 0, w: 4, h: 7 },
        { i: 'trades', x: 8, y: 7, w: 4, h: 7 },
        { i: 'assets', x: 0, y: 14, w: 4, h: 7 },
        { i: 'portfolio', x: 4, y: 14, w: 4, h: 7 },
        { i: 'controls', x: 8, y: 14, w: 4, h: 7 },
        { i: 'strategies', x: 0, y: 21, w: 4, h: 7 },
        { i: 'patterns', x: 4, y: 21, w: 4, h: 7 },
        { i: 'volume', x: 8, y: 21, w: 4, h: 7 },
        { i: 'regimes', x: 0, y: 28, w: 6, h: 7 },
        { i: 'signals', x: 6, y: 28, w: 6, h: 7 },
        { i: 'pnl', x: 0, y: 35, w: 6, h: 7 },
        { i: 'alerts', x: 6, y: 35, w: 6, h: 7 },
        { i: 'news', x: 0, y: 42, w: 6, h: 7 },
        { i: 'brain', x: 6, y: 42, w: 6, h: 7 },
        { i: 'voice', x: 0, y: 49, w: 12, h: 4 },
        { i: 'platform', x: 0, y: 53, w: 6, h: 4 },
        { i: 'status', x: 6, y: 53, w: 6, h: 4 },
        { i: 'correlation', x: 0, y: 57, w: 6, h: 7 },
        { i: 'risk', x: 6, y: 57, w: 6, h: 7 }
    ],
    sm: [
        { i: 'chart', x: 0, y: 0, w: 12, h: 14 },
        { i: 'orderbook', x: 0, y: 14, w: 6, h: 8 },
        { i: 'trades', x: 6, y: 14, w: 6, h: 8 },
        { i: 'assets', x: 0, y: 22, w: 6, h: 8 },
        { i: 'portfolio', x: 6, y: 22, w: 6, h: 8 },
        { i: 'controls', x: 0, y: 30, w: 12, h: 8 },
        { i: 'strategies', x: 0, y: 38, w: 12, h: 8 },
        { i: 'patterns', x: 0, y: 46, w: 12, h: 8 },
        { i: 'volume', x: 0, y: 54, w: 12, h: 8 },
        { i: 'regimes', x: 0, y: 62, w: 12, h: 8 },
        { i: 'signals', x: 0, y: 70, w: 12, h: 8 },
        { i: 'pnl', x: 0, y: 78, w: 12, h: 8 },
        { i: 'alerts', x: 0, y: 86, w: 12, h: 8 },
        { i: 'news', x: 0, y: 94, w: 12, h: 8 },
        { i: 'brain', x: 0, y: 102, w: 12, h: 8 },
        { i: 'voice', x: 0, y: 110, w: 12, h: 6 },
        { i: 'platform', x: 0, y: 116, w: 12, h: 6 },
        { i: 'status', x: 0, y: 122, w: 12, h: 6 },
        { i: 'correlation', x: 0, y: 128, w: 12, h: 8 },
        { i: 'risk', x: 0, y: 136, w: 12, h: 8 }
    ],
    xs: [
        { i: 'chart', x: 0, y: 0, w: 12, h: 14 },
        { i: 'orderbook', x: 0, y: 14, w: 12, h: 8 },
        { i: 'trades', x: 0, y: 22, w: 12, h: 8 },
        { i: 'assets', x: 0, y: 30, w: 12, h: 8 },
        { i: 'portfolio', x: 0, y: 38, w: 12, h: 8 },
        { i: 'controls', x: 0, y: 46, w: 12, h: 8 },
        { i: 'strategies', x: 0, y: 54, w: 12, h: 8 },
        { i: 'patterns', x: 0, y: 62, w: 12, h: 8 },
        { i: 'volume', x: 0, y: 70, w: 12, h: 8 },
        { i: 'regimes', x: 0, y: 78, w: 12, h: 8 },
        { i: 'signals', x: 0, y: 86, w: 12, h: 8 },
        { i: 'pnl', x: 0, y: 94, w: 12, h: 8 },
        { i: 'alerts', x: 0, y: 102, w: 12, h: 8 },
        { i: 'news', x: 0, y: 110, w: 12, h: 8 },
        { i: 'brain', x: 0, y: 118, w: 12, h: 8 },
        { i: 'voice', x: 0, y: 126, w: 12, h: 6 },
        { i: 'platform', x: 0, y: 132, w: 12, h: 6 },
        { i: 'status', x: 0, y: 138, w: 12, h: 6 },
        { i: 'correlation', x: 0, y: 144, w: 12, h: 8 },
        { i: 'risk', x: 0, y: 152, w: 12, h: 8 }
    ]
};

// Saved layout profiles for different trading styles
const layoutProfiles = {
    default: layouts,
    scalper: {
        lg: [
            { i: 'chart', x: 0, y: 0, w: 8, h: 16 },
            { i: 'orderbook', x: 8, y: 0, w: 4, h: 8 },
            { i: 'trades', x: 8, y: 8, w: 4, h: 8 },
            { i: 'controls', x: 0, y: 16, w: 4, h: 8 },
            { i: 'volume', x: 4, y: 16, w: 4, h: 8 },
            { i: 'signals', x: 8, y: 16, w: 4, h: 8 },
            { i: 'patterns', x: 0, y: 24, w: 4, h: 8 },
            { i: 'voice', x: 4, y: 24, w: 8, h: 4 },
            { i: 'status', x: 4, y: 28, w: 8, h: 4 }
        ],
        // md, sm, and xs would be defined similarly
    },
    swingTrader: {
        lg: [
            { i: 'chart', x: 0, y: 0, w: 12, h: 16 },
            { i: 'patterns', x: 0, y: 16, w: 4, h: 8 },
            { i: 'regimes', x: 4, y: 16, w: 4, h: 8 },
            { i: 'signals', x: 8, y: 16, w: 4, h: 8 },
            { i: 'news', x: 0, y: 24, w: 6, h: 8 },
            { i: 'brain', x: 6, y: 24, w: 6, h: 8 },
            { i: 'voice', x: 0, y: 32, w: 12, h: 4 },
        ],
        // md, sm, and xs would be defined similarly
    }
    // Additional profiles could be defined
};

const Dashboard = () => {
    const dispatch = useDispatch();
    
    // Redux state selectors
    const activeAsset = useSelector(state => state.trading.activeAsset);
    const currentPlatform = useSelector(state => state.trading.platform);
    const marketData = useSelector(state => state.market.data);
    const orderBook = useSelector(state => state.market.orderBook);
    const tradeHistory = useSelector(state => state.market.trades);
    const portfolio = useSelector(state => state.account.portfolio);
    const positions = useSelector(state => state.account.positions);
    const strategies = useSelector(state => state.strategies.available);
    const activeStrategies = useSelector(state => state.strategies.active);
    const patternDetections = useSelector(state => state.intelligence.patterns);
    const signals = useSelector(state => state.intelligence.signals);
    const alerts = useSelector(state => state.alerts.items);
    const news = useSelector(state => state.news.items);
    const systemStatus = useSelector(state => state.system.status);
    const brainActivity = useSelector(state => state.intelligence.brainActivity);
    const voiceEnabled = useSelector(state => state.settings.voiceAdvisorEnabled);
    const darkMode = useSelector(state => state.settings.darkMode);
    const availableTimeframes = useSelector(state => state.market.availableTimeframes);
    const selectedTimeframe = useSelector(state => state.market.selectedTimeframe);
    
    // Component state
    const [currentLayouts, setCurrentLayouts] = useState(layouts);
    const [selectedProfile, setSelectedProfile] = useState('default');
    const [isAutoTrading, setIsAutoTrading] = useState(false);
    const [savedLayouts, setSavedLayouts] = useState({});
    const [isPanelConfigOpen, setIsPanelConfigOpen] = useState(false);
    const [visiblePanels, setVisiblePanels] = useState({
        chart: true,
        orderbook: true,
        trades: true,
        assets: true,
        portfolio: true,
        controls: true,
        strategies: true,
        patterns: true,
        volume: true,
        regimes: true,
        signals: true,
        pnl: true,
        alerts: true,
        news: true,
        brain: true,
        voice: true,
        platform: true,
        status: true,
        correlation: true,
        risk: true
    });

    // Refs
    const layoutRef = useRef(null);

    // Initialize data on component mount
    useEffect(() => {
        // Fetch initial market data for the active asset
        dispatch(fetchMarketData(currentPlatform, activeAsset, selectedTimeframe));
        
        // Subscribe to real-time order book and trade data
        dispatch(subscribeToOrderBook(currentPlatform, activeAsset));
        dispatch(subscribeToTrades(currentPlatform, activeAsset));
        
        // Load saved layouts from localStorage if available
        const savedLayoutsStr = localStorage.getItem('quantumSpectre_savedLayouts');
        if (savedLayoutsStr) {
            try {
                const parsed = JSON.parse(savedLayoutsStr);
                setSavedLayouts(parsed);
            } catch (error) {
                console.error('Error loading saved layouts:', error);
            }
        }
        
        // Clean up subscriptions on unmount
        return () => {
            // Unsubscribe from all data feeds
            // Implementation would depend on how subscriptions are managed
        };
    }, []);
    
    // Handle platform or asset changes
    useEffect(() => {
        // Update data when platform or asset changes
        if (activeAsset && currentPlatform) {
            dispatch(fetchMarketData(currentPlatform, activeAsset, selectedTimeframe));
            dispatch(subscribeToOrderBook(currentPlatform, activeAsset));
            dispatch(subscribeToTrades(currentPlatform, activeAsset));
            
            // Start strategy simulation for this asset
            if (activeStrategies.length > 0) {
                dispatch(startStrategySimulation(currentPlatform, activeAsset, activeStrategies));
            }
        }
    }, [activeAsset, currentPlatform, selectedTimeframe]);
    
    // Handle layout changes
    const onLayoutChange = useCallback((layout, layouts) => {
        setCurrentLayouts(layouts);
    }, []);
    
    // Save current layout
    const saveCurrentLayout = useCallback((name) => {
        const newSavedLayouts = {
            ...savedLayouts,
            [name]: currentLayouts
        };
        setSavedLayouts(newSavedLayouts);
        localStorage.setItem('quantumSpectre_savedLayouts', JSON.stringify(newSavedLayouts));
    }, [savedLayouts, currentLayouts]);
    
    // Load a saved layout
    const loadLayout = useCallback((name) => {
        if (name === 'default') {
            setCurrentLayouts(layouts);
            setSelectedProfile('default');
        } else if (layoutProfiles[name]) {
            setCurrentLayouts(layoutProfiles[name]);
            setSelectedProfile(name);
        } else if (savedLayouts[name]) {
            setCurrentLayouts(savedLayouts[name]);
            setSelectedProfile(name);
        }
    }, [savedLayouts]);
    
    // Toggle auto-trading
    const toggleAutoTrading = useCallback(() => {
        setIsAutoTrading(prevState => !prevState);
        // Implement auto-trading logic here
    }, []);
    
    // Change active asset
    const handleAssetChange = useCallback((asset) => {
        dispatch(updateActiveAsset(asset));
    }, [dispatch]);
    
    // Switch between platforms
    const handlePlatformChange = useCallback((platform) => {
        dispatch(togglePlatform(platform));
    }, [dispatch]);
    
    // Toggle panel visibility
    const togglePanelVisibility = useCallback((panelId) => {
        setVisiblePanels(prev => ({
            ...prev,
            [panelId]: !prev[panelId]
        }));
    }, []);
    
    // Request voice advice
    const handleVoiceAdviceRequest = useCallback(() => {
        dispatch(requestVoiceAdvice(currentPlatform, activeAsset));
    }, [dispatch, currentPlatform, activeAsset]);
    
    // Change timeframe
    const handleTimeframeChange = useCallback((timeframe) => {
        dispatch(updateTimeframe(timeframe));
    }, [dispatch]);
    
    // Generate panel content based on panel ID
    const renderPanelContent = (panelId) => {
        switch(panelId) {
            case 'chart':
                return (
                    
                );
            case 'orderbook':
                return (
                    
                );
            case 'trades':
                return (
                    
                );
            case 'assets':
                return (
                    
                );
            case 'portfolio':
                return (
                    
                );
            case 'controls':
                return (
                    
                );
            case 'strategies':
                return (
                    
                );
            case 'patterns':
                return (
                    
                );
            case 'volume':
                return (
                    
                );
            case 'regimes':
                return (
                    
                );
            case 'signals':
                return (
                    
                );
            case 'pnl':
                return (
                    
                );
            case 'alerts':
                return (
                    
                );
            case 'news':
                return (
                    
                );
            case 'brain':
                return (
                    
                );
            case 'voice':
                return (
                    
                );
            case 'platform':
                return (
                    
                );
            case 'status':
                return (
                    
                );
            case 'correlation':
                return (
                    
                );
            case 'risk':
                return (
                    
                );
            default:
                return Unknown panel: {panelId};
        }
    };

    return (
        
            
                
                    QuantumSpectre Elite Trading
                    
                        {currentPlatform}
                        {activeAsset}
                        {marketData?.lastPrice && (
                            
                                {formatCurrency(marketData.lastPrice)}
                                = 0 ? 'positive' : 'negative'}`}>
                                    {formatPercentage(marketData.priceChangePercent)}
                                
                            
                        )}
                    
                
                
                
                    
                         loadLayout(e.target.value)}
                            className="layout-selector"
                        >
                            Default Layout
                            Scalping Layout
                            Swing Trading Layout
                            {Object.keys(savedLayouts).map(name => (
                                {name}
                            ))}
                        
                        
                         setIsPanelConfigOpen(!isPanelConfigOpen)}
                        >
                            
                            Configure Panels
                        
                        
                         {
                                const name = prompt('Enter a name for this layout:');
                                if (name) saveCurrentLayout(name);
                            }}
                        >
                            
                            Save Layout
                        
                    
                    
                    
                        
                            Auto-Trading
                            
                                
                                
                            
                        
                    
                
            
            
            {isPanelConfigOpen && (
                
                    Configure Visible Panels
                    
                        {Object.keys(visiblePanels).map(panelId => (
                            
                                
                                     togglePanelVisibility(panelId)}
                                    />
                                    {panelId.charAt(0).toUpperCase() + panelId.slice(1)}
                                
                            
                        ))}
                    
                     setIsPanelConfigOpen(false)}
                    >
                        Apply Changes
                    
                
            )}
            
            
                {Object.keys(visiblePanels).filter(panelId => visiblePanels[panelId]).map(panelId => (
                    
                        
                            {panelId.charAt(0).toUpperCase() + panelId.slice(1)}
                            
                                 togglePanelVisibility(panelId)}
                                >
                                    
                                
                            
                        
                        
                            {renderPanelContent(panelId)}
                        
                    
                ))}
            
        
    );
};

export default Dashboard;