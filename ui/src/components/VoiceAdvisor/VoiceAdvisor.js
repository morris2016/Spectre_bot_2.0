import React, { useState, useEffect, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { 
    FaMicrophone, 
    FaMicrophoneSlash, 
    FaVolumeUp, 
    FaVolumeMute,
    FaCog,
    FaInfoCircle
} from 'react-icons/fa';
import { 
    toggleVoiceEnabled, 
    updateVoiceSettings,
    triggerManualAdvice
} from '../../store/actions/voiceActions';
import { AI_VOICE_MODELS } from '../../constants/voice';
import './VoiceAdvisor.scss';

/**
 * VoiceAdvisor Component
 * 
 * Advanced AI-powered voice assistant that provides real-time trading advice,
 * signals, and insights. Features continuous learning capabilities and
 * customizable voice/personality settings.
 */
const VoiceAdvisor = () => {
    const dispatch = useDispatch();
    const audioRef = useRef(null);
    const speechSynthesis = window.speechSynthesis;
    
    // Redux state
    const { 
        enabled,
        settings,
        currentAdvice,
        advisoryHistory,
        isProcessing 
    } = useSelector(state => state.voice);
    
    const { 
        activeExchange,
        activeAsset,
        activeTimeframe,
        isAutoTrading
    } = useSelector(state => state.trading);
    
    // Local state
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [localSettings, setLocalSettings] = useState(settings);
    const [isSpeaking, setIsSpeaking] = useState(false);
    
    // Process advice changes and speak when new advice is received
    useEffect(() => {
        if (enabled && currentAdvice && currentAdvice.message && !isSpeaking) {
            speakAdvice(currentAdvice.message);
        }
    }, [currentAdvice, enabled]);
    
    // Cancel speech when disabled
    useEffect(() => {
        if (!enabled && isSpeaking) {
            speechSynthesis.cancel();
            setIsSpeaking(false);
        }
    }, [enabled]);
    
    // Initialize Speech Recognition API
    useEffect(() => {
        if (enabled && 'webkitSpeechRecognition' in window) {
            setupSpeechRecognition();
        }
        
        return () => {
            // Cleanup speech recognition on unmount
            speechSynthesis.cancel();
        };
    }, [enabled]);
    
    const setupSpeechRecognition = () => {
        const SpeechRecognition = window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = settings.language;
        
        recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
            
            // Process voice commands based on transcript
            processVoiceCommand(transcript.toLowerCase());
        };
        
        recognition.start();
    };
    
    const processVoiceCommand = (command) => {
        // Command patterns for trading operations
        if (command.includes('what should i do') || 
            command.includes('give me advice') || 
            command.includes('what's your analysis')) {
            dispatch(triggerManualAdvice());
        }
        
        // Add more command patterns as needed
    };
    
    const speakAdvice = (text) => {
        if (!enabled) return;
        
        // Cancel any ongoing speech
        speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Apply voice settings
        utterance.volume = settings.volume;
        utterance.rate = settings.speed;
        utterance.pitch = settings.pitch;
        
        // Select voice based on model
        const voices = speechSynthesis.getVoices();
        const selectedVoice = voices.find(voice => 
            voice.name === settings.voiceModel || 
            voice.name.includes('Google') || 
            voice.name.includes('Female')
        );
        
        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }
        
        // Set callbacks
        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = (e) => {
            console.error('Speech synthesis error:', e);
            setIsSpeaking(false);
        };
        
        // Speak the advice
        speechSynthesis.speak(utterance);
    };
    
    const handleToggleVoice = () => {
        dispatch(toggleVoiceEnabled(!enabled));
    };
    
    const handleSettingsChange = (key, value) => {
        setLocalSettings({
            ...localSettings,
            [key]: value
        });
    };
    
    const saveSettings = () => {
        dispatch(updateVoiceSettings(localSettings));
        setSettingsOpen(false);
    };
    
    const getAdviceTypeIcon = (type) => {
        switch (type) {
            case 'signal':
                return ;
            case 'pattern':
                return ;
            case 'insight':
                return ;
            case 'alert':
                return ;
            default:
                return ;
        }
    };
    
    return (
        
            
                AI Voice Advisor
                
                     setSettingsOpen(!settingsOpen)}
                        title="Voice Settings"
                    >
                        
                    
                    
                        {enabled ?  : }
                    
                
            
            
            {settingsOpen && (
                
                    Voice Settings
                    
                    
                        Voice Model:
                         handleSettingsChange('voiceModel', e.target.value)}
                        >
                            {AI_VOICE_MODELS.map(model => (
                                
                                    {model.name}
                                
                            ))}
                        
                    
                    
                    
                        Language:
                         handleSettingsChange('language', e.target.value)}
                        >
                            English (US)
                            English (UK)
                            Spanish
                            French
                            German
                            Japanese
                            Chinese
                        
                    
                    
                    
                        Volume: {localSettings.volume.toFixed(1)}
                         handleSettingsChange('volume', parseFloat(e.target.value))}
                        />
                    
                    
                    
                        Speed: {localSettings.speed.toFixed(1)}x
                         handleSettingsChange('speed', parseFloat(e.target.value))}
                        />
                    
                    
                    
                        Pitch: {localSettings.pitch.toFixed(1)}
                         handleSettingsChange('pitch', parseFloat(e.target.value))}
                        />
                    
                    
                    
                        Verbosity:
                         handleSettingsChange('verbosity', e.target.value)}
                        >
                            Concise
                            Normal
                            Detailed
                        
                    
                    
                    
                        Personality:
                         handleSettingsChange('personality', e.target.value)}
                        >
                            Professional
                            Friendly
                            Assertive
                            Educational
                        
                    
                    
                    
                         setSettingsOpen(false)}>Cancel
                        Save
                    
                
            )}
            
            
                {isProcessing ? (
                    
                        
                            
                            
                            
                            
                            
                        
                        Processing market data...
                    
                ) : currentAdvice ? (
                    
                        
                            {getAdviceTypeIcon(currentAdvice.type)}
                            {currentAdvice.type.toUpperCase()}
                            {new Date(currentAdvice.timestamp).toLocaleTimeString()}
                        
                        {currentAdvice.message}
                        
                            
                                Confidence: {currentAdvice.confidence}%
                            
                            
                                Source: {currentAdvice.source}
                            
                        
                    
                ) : (
                    
                        
                        No recent advice. AI advisor is {enabled ? 'monitoring the market' : 'disabled'}.
                    
                )}
            
            
            
                Recent Insights
                
                    {advisoryHistory.length > 0 ? (
                        advisoryHistory.map((advice, index) => (
                            
                                
                                    {getAdviceTypeIcon(advice.type)}
                                    
                                        {new Date(advice.timestamp).toLocaleTimeString()}
                                    
                                
                                {advice.message}
                            
                        ))
                    ) : (
                        No advice history yet.
                    )}
                
            
            
            
        
    );
};

export default VoiceAdvisor;