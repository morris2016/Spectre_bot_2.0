import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { 
    Button, 
    TextField, 
    Select, 
    MenuItem, 
    FormControl, 
    InputLabel, 
    Slider, 
    Typography, 
    Chip,
    Grid,
    Divider,
    Box,
    Switch,
    FormControlLabel,
    Badge,
    Tooltip,
    IconButton,
    Paper
} from '@material-ui/core';
import { 
    Timeline, 
    TimelineItem, 
    TimelineSeparator, 
    TimelineConnector, 
    TimelineContent, 
    TimelineDot
} from '@material-ui/lab';
import { 
    TrendingUp, 
    TrendingDown, 
    AutorenewOutlined, 
    FiberManualRecord, 
    WarningOutlined,
    ArrowUpward,
    ArrowDownward,
    InfoOutlined,
    HistoryOutlined,
    PauseCircleOutline,
    PlayCircleOutline,
    AttachMoneyOutlined,
    SettingsBackupRestoreOutlined,
    LoopOutlined
} from '@material-ui/icons';
import { Line } from 'react-chartjs-2';
import { format } from 'date-fns';
import { toast } from 'react-toastify';
import AnimatedNumber from 'react-animated-number';
import { useHotkeys } from 'react-hotkeys-hook';

import { submitOrder, cancelOrder, modifyOrder } from '../../actions/orderActions';
import { setRiskLevel, toggleAutomation } from '../../actions/settingActions';
import { usePlatformContext } from '../../contexts/PlatformContext';
import { useBrainSignals } from '../../hooks/useBrainSignals';
import { useOrderHistory } from '../../hooks/useOrderHistory';
import { RISK_LEVELS, ORDER_TYPES, ORDER_SIDES, TIME_IN_FORCE } from '../../constants/orderConstants';
import { calculatePositionSize, calculateRiskAmount } from '../../utils/riskCalculations';
import { formatCurrency, formatPercent } from '../../utils/formatters';

import './OrderPanel.scss';

/**
 * OrderPanel Component
 * 
 * Sophisticated order panel that provides comprehensive functionality for manual and automated order execution.
 * Features include risk management sliders, real-time brain suggestions, multi-timeframe support,
 * position sizing calculators, quick trade buttons, and order history.
 */
const OrderPanel = () => {
    const dispatch = useDispatch();
    const { activePlatform, switchPlatform } = usePlatformContext();
    
    // Redux state
    const { symbol, price, bidPrice, askPrice } = useSelector(state => state.market.activeSymbol);
    const { balance, positions } = useSelector(state => state.account);
    const { automated, riskLevel } = useSelector(state => state.settings);
    const { pendingOrders } = useSelector(state => state.orders);
    
    // Custom hooks
    const { signals, confidence, direction } = useBrainSignals(symbol);
    const { recentOrders } = useOrderHistory(symbol, 5);
    
    // Local state
    const [orderType, setOrderType] = useState(ORDER_TYPES.MARKET);
    const [orderSide, setOrderSide] = useState(ORDER_SIDES.BUY);
    const [quantity, setQuantity] = useState('');
    const [limitPrice, setLimitPrice] = useState('');
    const [stopPrice, setStopPrice] = useState('');
    const [timeInForce, setTimeInForce] = useState(TIME_IN_FORCE.GTC);
    const [riskPercent, setRiskPercent] = useState(riskLevel);
    const [positionSizing, setPositionSizing] = useState('fixed');
    const [orderSubmitting, setOrderSubmitting] = useState(false);

    // Get current position for this symbol if exists
    const currentPosition = useMemo(() => 
        positions.find(pos => pos.symbol === symbol), [positions, symbol]);
        
    // Calculate suggested position size based on risk
    const suggestedQuantity = useMemo(() => 
        calculatePositionSize(
            balance.available, 
            price, 
            riskPercent, 
            activePlatform,
            symbol
        ), [balance.available, price, riskPercent, activePlatform, symbol]);

    // Calculate risk amount
    const riskAmount = useMemo(() => 
        calculateRiskAmount(
            balance.available, 
            riskPercent
        ), [balance.available, riskPercent]);
        
    // Listen for hotkeys
    useHotkeys('shift+b', () => handleQuickOrder(ORDER_SIDES.BUY), [handleQuickOrder]);
    useHotkeys('shift+s', () => handleQuickOrder(ORDER_SIDES.SELL), [handleQuickOrder]);
    useHotkeys('shift+c', handleCancelAllOrders, [handleCancelAllOrders]);

    // Update local state when Redux risk level changes
    useEffect(() => {
        setRiskPercent(riskLevel);
    }, [riskLevel]);

    // Update limit price when market price changes
    useEffect(() => {
        if (price && orderType === ORDER_TYPES.LIMIT) {
            setLimitPrice(price.toFixed(getPriceDecimals()));
        }
    }, [price, orderType]);

    // Update suggested quantity when risk percent changes
    useEffect(() => {
        if (positionSizing === 'risk-based') {
            setQuantity(suggestedQuantity.toFixed(getQuantityDecimals()));
        }
    }, [suggestedQuantity, positionSizing]);

    // Get the number of decimal places for the price based on the symbol
    const getPriceDecimals = useCallback(() => {
        // This would come from symbol info in a real implementation
        return activePlatform === 'binance' ? 2 : 4;
    }, [activePlatform]);

    // Get the number of decimal places for quantity based on the symbol
    const getQuantityDecimals = useCallback(() => {
        // This would come from symbol info in a real implementation
        return activePlatform === 'binance' ? 6 : 2;
    }, [activePlatform]);

    // Handle order submission
    const handleSubmitOrder = useCallback(async () => {
        try {
            setOrderSubmitting(true);
            
            const orderPayload = {
                symbol,
                side: orderSide,
                type: orderType,
                quantity: parseFloat(quantity),
                timeInForce,
                platform: activePlatform
            };
            
            if (orderType === ORDER_TYPES.LIMIT) {
                orderPayload.price = parseFloat(limitPrice);
            }
            
            if (orderType === ORDER_TYPES.STOP || orderType === ORDER_TYPES.STOP_LIMIT) {
                orderPayload.stopPrice = parseFloat(stopPrice);
                if (orderType === ORDER_TYPES.STOP_LIMIT) {
                    orderPayload.price = parseFloat(limitPrice);
                }
            }
            
            await dispatch(submitOrder(orderPayload));
            toast.success(`${orderSide} order submitted successfully`);
            
            // Reset fields
            if (positionSizing === 'fixed') {
                setQuantity('');
            }
        } catch (error) {
            toast.error(`Order error: ${error.message}`);
        } finally {
            setOrderSubmitting(false);
        }
    }, [
        dispatch, symbol, orderSide, orderType, quantity, 
        timeInForce, limitPrice, stopPrice, activePlatform, positionSizing
    ]);
    
    // Handle quick market order (for one-click trading)
    const handleQuickOrder = useCallback((side) => {
        if (!suggestedQuantity) return;
        
        dispatch(submitOrder({
            symbol,
            side,
            type: ORDER_TYPES.MARKET,
            quantity: suggestedQuantity,
            platform: activePlatform
        }));
        
        toast.success(`Quick ${side} order executed`);
    }, [dispatch, symbol, suggestedQuantity, activePlatform]);
    
    // Handle cancel all pending orders
    const handleCancelAllOrders = useCallback(() => {
        const ordersForSymbol = pendingOrders.filter(
            order => order.symbol === symbol
        );
        
        ordersForSymbol.forEach(order => {
            dispatch(cancelOrder(order.id, activePlatform));
        });
        
        if (ordersForSymbol.length > 0) {
            toast.info(`Cancelled ${ordersForSymbol.length} pending orders`);
        }
    }, [dispatch, pendingOrders, symbol, activePlatform]);
    
    // Handle risk slider change
    const handleRiskChange = useCallback((event, newValue) => {
        setRiskPercent(newValue);
        dispatch(setRiskLevel(newValue));
    }, [dispatch]);
    
    // Handle automation toggle
    const handleAutomationToggle = useCallback(() => {
        dispatch(toggleAutomation(!automated));
        toast.info(`Trading automation ${!automated ? 'enabled' : 'disabled'}`);
    }, [dispatch, automated]);
    
    // Signal strength calculation
    const signalStrength = useMemo(() => {
        if (!confidence) return 0;
        return confidence * (direction === ORDER_SIDES.BUY ? 1 : -1);
    }, [confidence, direction]);
    
    // Signal color based on strength and direction
    const getSignalColor = useCallback((strength) => {
        const absStrength = Math.abs(strength);
        if (absStrength < 0.3) return '#9e9e9e';
        if (absStrength < 0.6) return strength > 0 ? '#4caf50' : '#ff9800';
        return strength > 0 ? '#00c853' : '#f44336';
    }, []);
    
    // Order book visualization data
    const orderBookData = useMemo(() => ({
        labels: ['-2%', '-1%', 'Price', '+1%', '+2%'],
        datasets: [{
            label: 'Buy Orders',
            data: [10, 25, 40, 12, 8],
            backgroundColor: 'rgba(76, 175, 80, 0.2)',
            borderColor: 'rgba(76, 175, 80, 1)',
            borderWidth: 1,
            pointRadius: 3
        },
        {
            label: 'Sell Orders',
            data: [5, 18, 35, 28, 15],
            backgroundColor: 'rgba(244, 67, 54, 0.2)',
            borderColor: 'rgba(244, 67, 54, 1)',
            borderWidth: 1,
            pointRadius: 3
        }]
    }), []);

    return (
        
            
                
                    
                        
                            
                                 switchPlatform(
                                        activePlatform === 'binance' ? 'deriv' : 'binance'
                                    )}
                                >
                                    {activePlatform === 'binance' ? 'B' : 'D'}
                                
                            
                        
                    

                    
                        {symbol} Order Panel
                    
                    
                    
                        }
                        label={
                            
                                {automated ? 'Auto' : 'Manual'}
                            
                        }
                    />
                
                
                
                
                
                    
                        
                            Current Price
                            
                                {formatCurrency(price)}
                            
                        
                        
                            Bid
                            
                                {formatCurrency(bidPrice)}
                            
                        
                        
                            Ask
                            
                                {formatCurrency(askPrice)}
                            
                        
                    
                
                
                {currentPosition && (
                    
                         : }
                            label={`${currentPosition.side} ${currentPosition.quantity} @ ${formatCurrency(currentPosition.entryPrice)}`}
                            color={currentPosition.side === 'BUY' ? 'primary' : 'secondary'}
                        />
                        
                            = 0 ? 'profit' : 'loss'}
                            >
                                P/L: {formatCurrency(currentPosition.unrealizedPnl)} 
                                ({formatPercent(currentPosition.unrealizedPnlPercent)})
                            
                        
                    
                )}
                
                
                    
                        
                             0.7 ? 'secondary' : 'default'} variant="dot">
                                Brain Signals
                            
                        
                         : }
                            label={`${(confidence * 100).toFixed(0)}% ${direction}`}
                            style={{ 
                                backgroundColor: getSignalColor(signalStrength),
                                color: Math.abs(signalStrength) > 0.5 ? 'white' : 'inherit'
                            }}
                        />
                    
                    
                    
                         0 ? 'auto' : 0
                            }}
                        >
                    
                    
                    {signals.length > 0 && (
                        
                            {signals.slice(0, 3).map((signal, idx) => (
                                
                                    
                                         0.7 ? 'default' : 'outlined'}
                                        >
                                            {signal.type === 'buy' ?  : }
                                        
                                        {idx < 2 && }
                                    
                                    
                                        
                                            {signal.message}
                                        
                                        
                                            {format(new Date(signal.timestamp), 'HH:mm:ss')}
                                        
                                    
                                
                            ))}
                        
                    )}
                
                
                
                    
                        Risk Management
                    
                    
                    
                        
                             `${x}%`}
                            />
                        
                        
                            
                                }
                                    label={formatCurrency(riskAmount)}
                                    color={riskPercent > 2 ? 'secondary' : 'default'}
                                    variant="outlined"
                                />
                            
                        
                    
                    
                     setPositionSizing(
                                    positionSizing === 'fixed' ? 'risk-based' : 'fixed'
                                )}
                                name="positionSizing"
                                color="primary"
                                size="small"
                            />
                        }
                        label={
                            
                                Risk-based sizing
                            
                        }
                    />
                
                
                
                    
                        
                            
                                Order Type
                                 setOrderType(e.target.value)}
                                    label="Order Type"
                                >
                                    Market
                                    Limit
                                    Stop
                                    Stop Limit
                                
                            
                        
                        
                        
                            
                                Time In Force
                                 setTimeInForce(e.target.value)}
                                    label="Time In Force"
                                >
                                    Good Till Cancel
                                    Immediate or Cancel
                                    Fill or Kill
                                
                            
                        
                        
                        
                             setQuantity(e.target.value)}
                                type="number"
                                size="small"
                                InputProps={{
                                    endAdornment: (
                                        
                                             setQuantity(
                                                    suggestedQuantity.toFixed(getQuantityDecimals())
                                                )}
                                            >
                                                
                                            
                                        
                                    )
                                }}
                            />
                        
                        
                        {(orderType === ORDER_TYPES.LIMIT || orderType === ORDER_TYPES.STOP_LIMIT) && (
                            
                                 setLimitPrice(e.target.value)}
                                    type="number"
                                    size="small"
                                />
                            
                        )}
                        
                        {(orderType === ORDER_TYPES.STOP || orderType === ORDER_TYPES.STOP_LIMIT) && (
                            
                                 setStopPrice(e.target.value)}
                                    type="number"
                                    size="small"
                                />
                            
                        )}
                    
                
                
                
                    
                        
                             {
                                    setOrderSide(ORDER_SIDES.BUY);
                                    handleSubmitOrder();
                                }}
                                disabled={orderSubmitting || !quantity}
                                startIcon={}
                                className="buy-button"
                            >
                                Buy / Long
                            
                        
                        
                        
                             {
                                    setOrderSide(ORDER_SIDES.SELL);
                                    handleSubmitOrder();
                                }}
                                disabled={orderSubmitting || !quantity}
                                startIcon={}
                                className="sell-button"
                            >
                                Sell / Short
                            
                        
                        
                        
                            
                                }
                                >
                                    Cancel All
                                
                                
                                
                                    {automated ?  : }
                                
                                
                                 {/* Toggle order history */}}
                                    title="Order History"
                                >
                                    
                                
                            
                        
                    
                
                
                {recentOrders.length > 0 && (
                    
                        
                            Recent Orders
                        
                        
                        
                            {recentOrders.map((order, idx) => (
                                
                                    
                                        
                                        {order.side}
                                    
                                    
                                        {order.quantity} @ {formatCurrency(order.price)}
                                    
                                    
                                        {format(new Date(order.time), 'HH:mm:ss')}
                                    
                                
                            ))}
                        
                    
                )}
                
                
                    
                        Order Book Depth
                    
                    
                    
                        
                    
                
            
        
    );
};

export default OrderPanel;