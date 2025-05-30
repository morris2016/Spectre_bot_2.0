import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  IconButton, 
  TextField, 
  Button, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemAvatar, 
  Avatar, 
  Divider, 
  CircularProgress,
  Tooltip,
  Fade,
  Zoom,
  Chip,
  Menu,
  MenuItem,
  Switch,
  FormControlLabel
} from '@mui/material';
import { 
  Mic, 
  MicOff, 
  Send, 
  Psychology, 
  Person, 
  VolumeUp, 
  VolumeMute,
  Settings,
  Delete,
  Save,
  ContentCopy,
  MoreVert,
  Refresh,
  Help
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { useVoiceAssistant } from '../../contexts/VoiceAdvisorContext';
import api from '../../api';

const VoiceAssistantPanel = () => {
  const theme = useTheme();
  const { 
    isListening, 
    toggleListening, 
    transcript, 
    clearTranscript,
    isSpeaking,
    toggleSpeaking,
    speak
  } = useVoiceAssistant();
  
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [settingsAnchorEl, setSettingsAnchorEl] = useState(null);
  const [voiceSettings, setVoiceSettings] = useState({
    autoListen: true,
    autoSpeak: true,
    voice: 'default',
    speed: 1.0,
    pitch: 1.0
  });
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Update input text when transcript changes
  useEffect(() => {
    if (transcript && isListening) {
      setInputText(transcript);
    }
  }, [transcript, isListening]);

  // Auto-submit when voice recognition stops
  useEffect(() => {
    if (transcript && !isListening && voiceSettings.autoListen) {
      handleSendMessage();
    }
  }, [isListening, transcript]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;
    
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputText,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    clearTranscript();
    setIsProcessing(true);
    setError(null);
    
    try {
      const response = await api.post('/api/voice-assistant/query', {
        query: userMessage.content,
        context: {
          recent_messages: messages.slice(-5),
          current_asset: 'BTCUSDT', // This would come from your app state
          current_timeframe: '1h'
        }
      });
      
      if (response.data && response.data.response) {
        const assistantMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: response.data.response,
          actions: response.data.actions || [],
          data: response.data.data || null,
          timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        
        // Auto-speak response if enabled
        if (voiceSettings.autoSpeak) {
          speak(assistantMessage.content);
        }
        
        // Execute any actions
        if (assistantMessage.actions && assistantMessage.actions.length > 0) {
          executeActions(assistantMessage.actions);
        }
      }
    } catch (err) {
      console.error('Error sending message to assistant:', err);
      setError('Failed to get a response. Please try again.');
    } finally {
      setIsProcessing(false);
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }
  };

  const executeActions = (actions) => {
    // Handle different action types
    actions.forEach(action => {
      switch (action.type) {
        case 'navigate':
          // This would integrate with your app's navigation
          console.log(`Navigate to: ${action.destination}`);
          break;
        case 'place_order':
          // This would integrate with your trading system
          console.log(`Place order: ${JSON.stringify(action.order)}`);
          break;
        case 'change_timeframe':
          console.log(`Change timeframe to: ${action.timeframe}`);
          break;
        case 'change_asset':
          console.log(`Change asset to: ${action.asset}`);
          break;
        default:
          console.log(`Unknown action type: ${action.type}`);
      }
    });
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearConversation = () => {
    setMessages([]);
    setSettingsAnchorEl(null);
  };

  const handleSaveConversation = () => {
    // This would save the conversation to your backend
    console.log('Saving conversation');
    setSettingsAnchorEl(null);
  };

  const handleCopyConversation = () => {
    const text = messages
      .map(msg => `${msg.role === 'user' ? 'You' : 'Assistant'}: ${msg.content}`)
      .join('\n\n');
    
    navigator.clipboard.writeText(text);
    setSettingsAnchorEl(null);
  };

  const handleSettingsClick = (event) => {
    setSettingsAnchorEl(event.currentTarget);
  };

  const handleSettingsClose = () => {
    setSettingsAnchorEl(null);
  };

  const handleVoiceSettingChange = (setting, value) => {
    setVoiceSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderMessageContent = (message) => {
    // Render different content types
    if (message.data && message.data.type === 'chart') {
      return (
        <Box mt={1}>
          <Typography variant="body2">{message.content}</Typography>
          <Box 
            mt={1} 
            p={1} 
            border={1} 
            borderColor="divider" 
            borderRadius={1}
            bgcolor={theme.palette.background.paper}
          >
            <Typography variant="caption" color="textSecondary">
              [Chart visualization would appear here]
            </Typography>
          </Box>
        </Box>
      );
    }
    
    if (message.data && message.data.type === 'table') {
      return (
        <Box mt={1}>
          <Typography variant="body2">{message.content}</Typography>
          <Box 
            mt={1} 
            p={1} 
            border={1} 
            borderColor="divider" 
            borderRadius={1}
            bgcolor={theme.palette.background.paper}
          >
            <Typography variant="caption" color="textSecondary">
              [Table data would appear here]
            </Typography>
          </Box>
        </Box>
      );
    }
    
    return <Typography variant="body2">{message.content}</Typography>;
  };

  return (
    <Paper elevation={0} variant="outlined">
      <Box p={2} display="flex" justifyContent="space-between" alignItems="center">
        <Box display="flex" alignItems="center">
          <Psychology color="primary" style={{ marginRight: theme.spacing(1) }} />
          <Typography variant="h6">Trading Assistant</Typography>
        </Box>
        
        <Box>
          <Tooltip title="Voice Settings">
            <IconButton onClick={handleSettingsClick} size="small">
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      <Divider />
      
      <Box 
        p={2} 
        sx={{ 
          height: 400, 
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: theme.palette.divider,
            borderRadius: '4px',
          }
        }}
      >
        {messages.length === 0 ? (
          <Box 
            display="flex" 
            flexDirection="column" 
            alignItems="center" 
            justifyContent="center" 
            flexGrow={1}
            p={3}
          >
            <Psychology style={{ fontSize: 60, color: theme.palette.text.secondary, opacity: 0.5 }} />
            <Typography variant="body1" color="textSecondary" align="center" mt={2}>
              Ask me anything about trading, market analysis, or your portfolio.
            </Typography>
            <Box mt={2}>
              <Chip 
                label="Show me BTC price prediction" 
                onClick={() => setInputText("Show me BTC price prediction")}
                clickable
              />
              <Chip 
                label="What's my portfolio performance?" 
                onClick={() => setInputText("What's my portfolio performance?")}
                clickable
                style={{ marginLeft: theme.spacing(1) }}
              />
              <Chip 
                label="Explain the current market regime" 
                onClick={() => setInputText("Explain the current market regime")}
                clickable
                style={{ marginTop: theme.spacing(1) }}
              />
            </Box>
          </Box>
        ) : (
          <List>
            {messages.map((message) => (
              <ListItem
                key={message.id}
                alignItems="flex-start"
                style={{ 
                  flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                  padding: theme.spacing(1, 0)
                }}
              >
                <ListItemAvatar>
                  <Avatar 
                    style={{ 
                      backgroundColor: message.role === 'user' 
                        ? theme.palette.primary.main 
                        : theme.palette.secondary.main 
                    }}
                  >
                    {message.role === 'user' ? <Person /> : <Psychology />}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Box 
                      display="flex" 
                      justifyContent={message.role === 'user' ? 'flex-end' : 'flex-start'}
                      alignItems="center"
                    >
                      <Typography 
                        variant="subtitle2" 
                        color="textPrimary"
                        style={{ marginRight: message.role === 'user' ? 0 : theme.spacing(1) }}
                      >
                        {message.role === 'user' ? 'You' : 'Assistant'}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {formatTimestamp(message.timestamp)}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Box 
                      mt={1} 
                      p={2} 
                      bgcolor={message.role === 'user' 
                        ? theme.palette.primary.light 
                        : theme.palette.background.paper
                      }
                      color={message.role === 'user' 
                        ? theme.palette.primary.contrastText 
                        : theme.palette.text.primary
                      }
                      borderRadius={2}
                      boxShadow={1}
                      style={{ 
                        textAlign: message.role === 'user' ? 'right' : 'left',
                        maxWidth: '80%',
                        marginLeft: message.role === 'user' ? 'auto' : 0,
                        marginRight: message.role === 'user' ? 0 : 'auto',
                      }}
                    >
                      {renderMessageContent(message)}
                      
                      {message.actions && message.actions.length > 0 && (
                        <Box mt={2} display="flex" flexWrap="wrap" gap={1}>
                          {message.actions.map((action, index) => (
                            <Chip
                              key={index}
                              label={action.label || action.type}
                              color="primary"
                              size="small"
                              clickable
                              onClick={() => executeActions([action])}
                            />
                          ))}
                        </Box>
                      )}
                    </Box>
                  }
                  style={{ margin: 0 }}
                />
              </ListItem>
            ))}
            <div ref={messagesEndRef} />
          </List>
        )}
      </Box>
      
      <Divider />
      
      <Box p={2}>
        {error && (
          <Typography color="error" variant="body2" gutterBottom>
            {error}
          </Typography>
        )}
        
        <Box display="flex" alignItems="center">
          <Tooltip title={isListening ? "Stop Listening" : "Start Listening"}>
            <IconButton 
              onClick={toggleListening} 
              color={isListening ? "secondary" : "default"}
              style={{ marginRight: theme.spacing(1) }}
            >
              {isListening ? <MicOff /> : <Mic />}
            </IconButton>
          </Tooltip>
          
          <TextField
            fullWidth
            placeholder="Ask me anything..."
            variant="outlined"
            value={inputText}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            disabled={isProcessing}
            inputRef={inputRef}
            InputProps={{
              endAdornment: (
                <Box display="flex" alignItems="center">
                  {isProcessing && (
                    <CircularProgress size={24} style={{ marginRight: theme.spacing(1) }} />
                  )}
                  <Tooltip title="Send Message">
                    <IconButton 
                      onClick={handleSendMessage} 
                      disabled={!inputText.trim() || isProcessing}
                      color="primary"
                    >
                      <Send />
                    </IconButton>
                  </Tooltip>
                </Box>
              ),
            }}
          />
          
          <Tooltip title={isSpeaking ? "Stop Speaking" : "Enable Speech"}>
            <IconButton 
              onClick={toggleSpeaking} 
              color={isSpeaking ? "primary" : "default"}
              style={{ marginLeft: theme.spacing(1) }}
            >
              {isSpeaking ? <VolumeUp /> : <VolumeMute />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      <Menu
        anchorEl={settingsAnchorEl}
        open={Boolean(settingsAnchorEl)}
        onClose={handleSettingsClose}
      >
        <MenuItem>
          <FormControlLabel
            control={
              <Switch
                checked={voiceSettings.autoListen}
                onChange={(e) => handleVoiceSettingChange('autoListen', e.target.checked)}
                color="primary"
              />
            }
            label="Auto-submit voice input"
          />
        </MenuItem>
        <MenuItem>
          <FormControlLabel
            control={
              <Switch
                checked={voiceSettings.autoSpeak}
                onChange={(e) => handleVoiceSettingChange('autoSpeak', e.target.checked)}
                color="primary"
              />
            }
            label="Auto-speak responses"
          />
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleClearConversation}>
          <ListItemText primary="Clear conversation" />
          <Delete fontSize="small" />
        </MenuItem>
        <MenuItem onClick={handleSaveConversation}>
          <ListItemText primary="Save conversation" />
          <Save fontSize="small" />
        </MenuItem>
        <MenuItem onClick={handleCopyConversation}>
          <ListItemText primary="Copy to clipboard" />
          <ContentCopy fontSize="small" />
        </MenuItem>
      </Menu>
    </Paper>
  );
};

export default VoiceAssistantPanel;