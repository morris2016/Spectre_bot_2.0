#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Deep Learning Models Module

This module implements advanced deep learning models for the QuantumSpectre Elite Trading System,
including LSTMs, GRUs, CNNs, and Transformer-based architectures optimized for financial time series.
The models are designed to be efficient on consumer-grade hardware (RTX 3050) while delivering
exceptional predictive performance.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import time
from datetime import datetime, timedelta
import pickle
import json
import warnings
from dataclasses import dataclass
from enum import Enum, auto

# Internal imports
from common.logger import get_logger
from common.exceptions import ModelError, DataError, TrainingError
from common.utils import timeit
from ml_models.hardware.gpu import setup_gpu, get_gpu_memory_usage
from ml_models.models.base import BaseModel, ModelConfig, ModelOutput, DataBatch

logger = get_logger(__name__)

# Enum for model types
class DeepModelType(Enum):
    LSTM = auto()
    GRU = auto()
    CONV1D = auto()
    TCNN = auto()  # Temporal CNN
    TRANSFORMER = auto()
    LSTM_ATTENTION = auto()
    GRU_ATTENTION = auto()
    WAVENET = auto()
    INCEPTIONTIME = auto()
    HYBRID = auto()

@dataclass
class DeepLearningConfig(ModelConfig):
    """Configuration for deep learning models"""
    model_type: DeepModelType = DeepModelType.LSTM
    input_dim: int = 0
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    batch_size: int = 64
    sequence_length: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_heads: int = 8  # For transformer models
    num_filters: int = 64  # For CNN models
    kernel_sizes: List[int] = None  # For CNN models
    attention_size: int = 64  # For attention-based models
    early_stopping_patience: int = 10
    use_mixed_precision: bool = True
    use_checkpoint: bool = False
    device: str = None
    clip_grad_norm: float = 1.0
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_steps: int = 100
    activation: str = "relu"
    initialization: str = "xavier"
    residual_connections: bool = True
    embed_dim: int = 64  # For embedding layers
    
    def __post_init__(self):
        """Validate configuration and set defaults"""
        super().__post_init__()
        
        # Set default kernel sizes for CNN models if not provided
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7]
            
        # Determine device if not explicitly set
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Validate model type specific parameters
        if self.model_type in [DeepModelType.TRANSFORMER, DeepModelType.LSTM_ATTENTION, 
                               DeepModelType.GRU_ATTENTION] and self.num_heads <= 0:
            raise ValueError(f"Number of attention heads must be positive, got {self.num_heads}")
            
        if self.model_type in [DeepModelType.CONV1D, DeepModelType.TCNN, DeepModelType.WAVENET, 
                               DeepModelType.INCEPTIONTIME] and self.num_filters <= 0:
            raise ValueError(f"Number of filters must be positive, got {self.num_filters}")
            
        # Input validation
        if self.input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {self.input_dim}")
            
        if self.sequence_length <= 0:
            raise ValueError(f"Sequence length must be positive, got {self.sequence_length}")
            
class FinancialTimeSeriesDataset(Dataset):
    """Custom dataset for financial time series data"""
    
    def __init__(self, 
                 features: np.ndarray, 
                 targets: np.ndarray, 
                 sequence_length: int,
                 device: str = "cpu"):
        """
        Initialize the dataset
        
        Args:
            features: Array of features [n_samples, n_features]
            targets: Array of targets [n_samples, n_targets]
            sequence_length: Length of the sequence for each sample
            device: Device to store the tensors on
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.device = device
        
        # Validate inputs
        if len(self.features) != len(self.targets):
            raise DataError("Features and targets must have the same length")
            
        if len(self.features) <= self.sequence_length:
            raise DataError("Not enough samples for the given sequence length")
            
    def __len__(self) -> int:
        """Return the number of samples"""
        return len(self.features) - self.sequence_length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset"""
        # Get sequence of features and corresponding target
        feature_seq = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length]
        
        # Convert to tensors and move to device
        feature_tensor = torch.tensor(feature_seq, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target, dtype=torch.float32).to(self.device)
        
        return feature_tensor, target_tensor

class LSTMModel(nn.Module):
    """LSTM model for financial time series prediction"""
    
    def __init__(self, config: DeepLearningConfig):
        """
        Initialize the LSTM model
        
        Args:
            config: Model configuration
        """
        super(LSTMModel, self).__init__()
        self.config = config
        
        # Input embedding layer to transform raw features
        self.embedding = nn.Linear(config.input_dim, config.embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.output_layer = nn.Linear(lstm_output_dim, config.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.initialization == "xavier":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif self.config.initialization == "kaiming":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Embed the input
        x = self.embedding(x)
        
        # Apply LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Get the output of the last time step
        if self.config.bidirectional:
            # Concatenate the last hidden state from both directions
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # Get the last hidden state
            hidden = hidden[-1]
            
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Apply output layer
        output = self.output_layer(hidden)
        
        return output

class GRUModel(nn.Module):
    """GRU model for financial time series prediction"""
    
    def __init__(self, config: DeepLearningConfig):
        """
        Initialize the GRU model
        
        Args:
            config: Model configuration
        """
        super(GRUModel, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Linear(config.input_dim, config.embed_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        gru_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.output_layer = nn.Linear(gru_output_dim, config.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.initialization == "xavier":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif self.config.initialization == "kaiming":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Embed the input
        x = self.embedding(x)
        
        # Apply GRU
        _, hidden = self.gru(x)
        
        # Get the output of the last time step
        if self.config.bidirectional:
            # Concatenate the last hidden state from both directions
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # Get the last hidden state
            hidden = hidden[-1]
            
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Apply output layer
        output = self.output_layer(hidden)
        
        return output

class Conv1DModel(nn.Module):
    """1D CNN model for financial time series prediction"""
    
    def __init__(self, config: DeepLearningConfig):
        """
        Initialize the 1D CNN model
        
        Args:
            config: Model configuration
        """
        super(Conv1DModel, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Linear(config.input_dim, config.embed_dim)
        
        # Convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embed_dim,
                out_channels=config.num_filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2  # Same padding
            )
            for kernel_size in config.kernel_sizes
        ])
        
        # Pooling layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Activation function
        self.activation = self._get_activation()
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        conv_output_dim = config.num_filters * len(config.kernel_sizes)
        self.output_layer = nn.Linear(conv_output_dim, config.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _get_activation(self):
        """Get activation function based on config"""
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif self.config.activation == "elu":
            return nn.ELU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU()
        
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.initialization == "xavier":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif self.config.initialization == "kaiming":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Embed the input
        x = self.embedding(x)
        
        # Transpose for conv1d layers which expect [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x)
            conv_out = self.activation(conv_out)
            pooled = self.pool(conv_out).squeeze(-1)
            conv_outputs.append(pooled)
            
        # Concatenate outputs from different kernel sizes
        x = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply output layer
        output = self.output_layer(x)
        
        return output

class SelfAttention(nn.Module):
    """Self-attention module for sequence models"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize the self-attention module
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Check if embed_dim is divisible by num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"Embedding dimension {embed_dim} must be divisible by number of heads {num_heads}")
            
        # Projection layers
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Get projections
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention weights to values
        x = torch.matmul(attention, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back to original dimensions
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, self.embed_dim)  # [batch_size, seq_len, embed_dim]
        
        # Apply output projection
        x = self.output_proj(x)
        
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize the transformer encoder layer
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Self attention
        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Self attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff_network(x)
        x = self.norm2(x + ff_output)
        
        return x

class TransformerModel(nn.Module):
    """Transformer model for financial time series prediction"""
    
    def __init__(self, config: DeepLearningConfig):
        """
        Initialize the transformer model
        
        Args:
            config: Model configuration
        """
        super(TransformerModel, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Linear(config.input_dim, config.embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.embed_dim, config.dropout, config.sequence_length)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.hidden_dim,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        self.output_layer = nn.Linear(config.embed_dim, config.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.initialization == "xavier":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif self.config.initialization == "kaiming":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Embed the input
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Get the output of the last time step
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply output layer
        output = self.output_layer(x)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the positional encoding
        
        Args:
            embed_dim: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        
        # Calculate positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LSTMAttentionModel(nn.Module):
    """LSTM with attention model for financial time series prediction"""
    
    def __init__(self, config: DeepLearningConfig):
        """
        Initialize the LSTM with attention model
        
        Args:
            config: Model configuration
        """
        super(LSTMAttentionModel, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Linear(config.input_dim, config.embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Attention layer
        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.attention = SelfAttention(lstm_output_dim, config.num_heads, config.dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(lstm_output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        self.output_layer = nn.Linear(lstm_output_dim, config.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.initialization == "xavier":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif self.config.initialization == "kaiming":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Embed the input
        x = self.embedding(x)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out = self.attention(lstm_out)
        
        # Residual connection and normalization
        x = self.norm(lstm_out + attn_out)
        
        # Get the output of the last time step
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply output layer
        output = self.output_layer(x)
        
        return output

class WaveNetBlock(nn.Module):
    """WaveNet residual block"""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        """
        Initialize the WaveNet block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dilation: Dilation factor
        """
        super(WaveNetBlock, self).__init__()
        
        # Dilated convolution
        self.dilated_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,  # For gated activation
            kernel_size=2,
            padding=dilation,
            dilation=dilation
        )
        
        # 1x1 convolution for residual connection
        self.res_conv = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
        # 1x1 convolution for skip connection
        self.skip_conv = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, seq_len]
            
        Returns:
            Tuple of (residual output, skip output) tensors
        """
        # Apply dilated convolution
        conv_out = self.dilated_conv(x)
        
        # Split for gated activation (filter and gate)
        filter_out, gate_out = torch.chunk(conv_out, 2, dim=1)
        
        # Apply gated activation
        gated_out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        
        # Apply residual and skip connections
        res_out = self.res_conv(gated_out)
        skip_out = self.skip_conv(gated_out)
        
        # Add residual connection
        res_out = res_out + x[:, :, -res_out.size(2):]
        
        return res_out, skip_out

class WaveNetModel(nn.Module):
    """WaveNet model for financial time series prediction"""
    
    def __init__(self, config: DeepLearningConfig):
        """
        Initialize the WaveNet model
        
        Args:
            config: Model configuration
        """
        super(WaveNetModel, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Linear(config.input_dim, config.embed_dim)
        
        # Initial causal convolution
        self.causal_conv = nn.Conv1d(
            in_channels=config.embed_dim,
            out_channels=config.num_filters,
            kernel_size=2,
            padding=1
        )
        
        # WaveNet blocks with exponentially increasing dilation
        self.wavenet_blocks = nn.ModuleList([
            WaveNetBlock(
                in_channels=config.num_filters,
                out_channels=config.num_filters,
                dilation=2**i
            )
            for i in range(config.num_layers)
        ])
        
        # 1x1 convolution layers after skip connections
        self.post_skip = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters, 1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters, 1)
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        self.output_layer = nn.Linear(config.num_filters, config.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.initialization == "xavier":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif self.config.initialization == "kaiming":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Embed the input
        x = self.embedding(x)
        
        # Transpose for conv1d layers [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply causal convolution
        x = self.causal_conv(x)
        
        # Apply WaveNet blocks
        skip_outputs = []
        for block in self.wavenet_blocks:
            x, skip = block(x)
            skip_outputs.append(skip)
            
        # Sum skip connections
        x = torch.stack(skip_outputs).sum(dim=0)
        
        # Apply post-skip processing
        x = self.post_skip(x)
        
        # Global average pooling
        x = x.mean(dim=2)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply output layer
        output = self.output_layer(x)
        
        return output

class InceptionBlock(nn.Module):
    """Inception block for time series"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int], bottleneck_size: int = None):
        """
        Initialize the inception block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per branch
            kernel_sizes: List of kernel sizes for each branch
            bottleneck_size: Optional bottleneck size for dimensionality reduction
        """
        super(InceptionBlock, self).__init__()
        
        self.bottleneck = None
        if bottleneck_size is not None:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1)
            in_channels = bottleneck_size
            
        # Create branches with different kernel sizes
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=(k - 1) // 2  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            for k in kernel_sizes
        ])
        
        # Bottleneck for maxpool branch
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # Residual connection if input and output dimensions match
        self.use_residual = (in_channels == out_channels * (len(kernel_sizes) + 1))
        
        if not self.use_residual:
            self.residual_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * (len(kernel_sizes) + 1), kernel_size=1),
                nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, out_channels*(len(kernel_sizes)+1), seq_len]
        """
        if self.bottleneck is not None:
            x = self.bottleneck(x)
            
        # Apply each branch
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Apply maxpool branch
        branch_outputs.append(self.maxpool_branch(x))
        
        # Concatenate branch outputs
        out = torch.cat(branch_outputs, dim=1)
        
        # Apply residual connection if dimensions match
        if self.use_residual:
            out = out + x
        elif hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
            
        return out

class InceptionTimeModel(nn.Module):
    """InceptionTime model for financial time series prediction"""
    
    def __init__(self, config: DeepLearningConfig):
        """
        Initialize the InceptionTime model
        
        Args:
            config: Model configuration
        """
        super(InceptionTimeModel, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Linear(config.input_dim, config.embed_dim)
        
        # Inception blocks
        self.blocks = nn.ModuleList()
        in_channels = config.embed_dim
        
        # Set default kernel sizes if not provided
        kernel_sizes = config.kernel_sizes if config.kernel_sizes else [9, 19, 39]
        
        # Add inception blocks
        for _ in range(config.num_layers):
            self.blocks.append(
                InceptionBlock(
                    in_channels=in_channels,
                    out_channels=config.num_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_size=config.num_filters // 2 if in_channels > config.num_filters else None
                )
            )
            in_channels = config.num_filters * (len(kernel_sizes) + 1)
            
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        self.output_layer = nn.Linear(in_channels, config.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.initialization == "xavier":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif self.config.initialization == "kaiming":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Embed the input
        x = self.embedding(x)
        
        # Transpose for conv1d layers [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply inception blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply global average pooling
        x = self.gap(x).squeeze(-1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply output layer
        output = self.output_layer(x)
        
        return output

class HybridModel(nn.Module):
    """Hybrid model combining multiple model types for financial time series prediction"""
    
    def __init__(self, config: DeepLearningConfig):
        """
        Initialize the hybrid model
        
        Args:
            config: Model configuration
        """
        super(HybridModel, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Linear(config.input_dim, config.embed_dim)
        
        # LSTM component
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if 2 > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # CNN component
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embed_dim,
                out_channels=config.num_filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2  # Same padding
            )
            for kernel_size in config.kernel_sizes
        ])
        
        # Attention component
        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.attention = SelfAttention(lstm_output_dim, config.num_heads, config.dropout)
        
        # Pooling layers
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(config.dropout)
        
        # Calculate output dimensions of each component
        conv_output_dim = config.num_filters * len(config.kernel_sizes)
        
        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim + conv_output_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.initialization == "xavier":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif self.config.initialization == "kaiming":
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Embed the input
        embedded = self.embedding(x)
        
        # LSTM component
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Apply attention to LSTM output
        attn_out = self.attention(lstm_out)
        
        # Get the output of the last time step for LSTM
        if self.config.bidirectional:
            # Concatenate the last hidden state from both directions
            lstm_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # Get the last hidden state
            lstm_final = hidden[-1]
            
        # CNN component
        # Transpose for conv1d layers [batch, channels, seq_len]
        conv_input = embedded.transpose(1, 2)
        
        # Apply convolutional layers
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(conv_input)
            conv_out = self.activation(conv_out)
            # Apply both max and average pooling
            max_pooled = self.max_pool(conv_out).squeeze(-1)
            conv_outputs.append(max_pooled)
            
        # Concatenate CNN outputs
        conv_final = torch.cat(conv_outputs, dim=1)
        
        # Concatenate LSTM and CNN outputs
        combined = torch.cat([lstm_final, conv_final], dim=1)
        
        # Apply dropout
        combined = self.dropout_layer(combined)
        
        # Apply final fully connected layers
        output = self.fc(combined)
        
        return output

class DeepLearningModel(BaseModel):
    """Deep learning model wrapper for the QuantumSpectre Elite Trading System"""
    
    def __init__(self, config: DeepLearningConfig, name: str = "deep_learning", **kwargs: Any):
        """
        Initialize the deep learning model
        
        Args:
            config: Model configuration
        """
        super(DeepLearningModel, self).__init__(config, name=name, **kwargs)
        self.config = config
        
        # Create model based on type
        self.model = self._build_model()
        
        # Move model to device
        self.model = self.model.to(config.device)
        
        # Set up optimizer
        self.optimizer = self._build_optimizer()
        
        # Set up learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Set up loss function
        self.criterion = nn.MSELoss()
        
        # Scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        logger.info(f"Initialized {config.model_type.name} model with {self._count_parameters()} parameters")
        
    def _build_model(self) -> nn.Module:
        """Build the model based on configuration"""
        if self.config.model_type == DeepModelType.LSTM:
            return LSTMModel(self.config)
        elif self.config.model_type == DeepModelType.GRU:
            return GRUModel(self.config)
        elif self.config.model_type == DeepModelType.CONV1D:
            return Conv1DModel(self.config)
        elif self.config.model_type == DeepModelType.TRANSFORMER:
            return TransformerModel(self.config)
        elif self.config.model_type == DeepModelType.LSTM_ATTENTION:
            return LSTMAttentionModel(self.config)
        elif self.config.model_type == DeepModelType.WAVENET:
            return WaveNetModel(self.config)
        elif self.config.model_type == DeepModelType.INCEPTIONTIME:
            return InceptionTimeModel(self.config)
        elif self.config.model_type == DeepModelType.HYBRID:
            return HybridModel(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer based on configuration"""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler based on configuration"""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=10
            )
        elif self.config.scheduler.lower() == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.config.scheduler.lower() == "one_cycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * 10,
                steps_per_epoch=100,  # Will be overridden in train
                epochs=10  # Will be overridden in train
            )
        return None
            
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            verbose: bool = True,
            callbacks: List[Callable] = None) -> Dict[str, List[float]]:
        """
        Fit the model to the training data
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            verbose: Whether to print progress
            callbacks: List of callback functions to call after each epoch
            
        Returns:
            Dictionary of training history
        """
        # Create datasets and data loaders
        train_dataset = FinancialTimeSeriesDataset(
            features=X_train,
            targets=y_train,
            sequence_length=self.config.sequence_length,
            device=self.config.device
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Adjust based on your system
            pin_memory=False
        )
        
        # Create validation loader if validation data is provided
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = FinancialTimeSeriesDataset(
                features=X_val,
                targets=y_val,
                sequence_length=self.config.sequence_length,
                device=self.config.device
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,  # Adjust based on your system
                pin_memory=False
            )
            
        # Update one-cycle scheduler if being used
        if self.config.scheduler.lower() == "one_cycle" and self.scheduler is not None:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * 10,
                steps_per_epoch=len(train_loader),
                epochs=epochs
            )
            
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move data to device
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                if self.scaler is not None:
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        
                    # Scale loss and calculate gradients
                    self.scaler.scale(loss).backward()
                    
                    # Clip gradients if needed
                    if self.config.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                        
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular training
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Calculate gradients
                    loss.backward()
                    
                    # Clip gradients if needed
                    if self.config.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                        
                    # Update weights
                    self.optimizer.step()
                    
                # Accumulate batch loss
                train_loss += loss.item()
                
                # Update learning rate scheduler if batch-based
                if self.scheduler is not None and isinstance(self.scheduler, 
                                                            (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
                    self.scheduler.step()
                    
                # Print progress if verbose
                if verbose and (batch_idx % 10 == 0):
                    logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                               f"Loss: {loss.item():.6f}")
                    
            # Average training loss
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase if validation data is provided
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                
                with torch.no_grad():
                    for data, target in val_loader:
                        # Move data to device
                        data, target = data.to(self.config.device), target.to(self.config.device)
                        
                        # Forward pass
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        
                        # Accumulate batch loss
                        val_loss += loss.item()
                        
                # Average validation loss
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    self.save(f"{self.config.model_name}_best.pt")
                else:
                    self.epochs_without_improvement += 1
                    
                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                # If no validation data, save model at each epoch
                self.save(f"{self.config.model_name}_epoch_{epoch+1}.pt")
                
            # Update learning rate scheduler if epoch-based
            if self.scheduler is not None and not isinstance(self.scheduler, 
                                                           (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader is not None else train_loss)
                else:
                    self.scheduler.step()
                    
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Print epoch results
            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | "
                           f"Train Loss: {train_loss:.6f} | "
                           f"Val Loss: {val_loss:.6f} | "
                           f"LR: {lr:.6f}")
                
            # Call callbacks if provided
            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch, {
                        'train_loss': train_loss,
                        'val_loss': val_loss if val_loader is not None else None,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                    
        # Return training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses if val_loader is not None else None
        }
        
        return history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        # Create dataset and data loader
        dataset = FinancialTimeSeriesDataset(
            features=X,
            targets=np.zeros((len(X), self.config.output_dim)),  # Dummy targets
            sequence_length=self.config.sequence_length,
            device=self.config.device
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Switch model to evaluation mode
        self.model.eval()
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for data, _ in loader:
                # Move data to device
                data = data.to(self.config.device)
                
                # Forward pass
                output = self.model(data)
                
                # Move predictions to CPU and convert to numpy
                predictions.append(output.cpu().numpy())
                
        # Concatenate predictions
        predictions = np.concatenate(predictions, axis=0)
        
        return predictions
        
    def save(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save the model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save model
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found")
            
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.config.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if exists
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        logger.info(f"Model loaded from {path}")
        
    def train_batch(self, batch: DataBatch) -> Dict[str, float]:
        """
        Train the model on a single batch
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of metrics
        """
        # Unpack batch
        data, target = batch
        
        # Move data to device
        data, target = data.to(self.config.device), target.to(self.config.device)
        
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
                
            # Scale loss and calculate gradients
            self.scaler.scale(loss).backward()
            
            # Clip gradients if needed
            if self.config.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                
            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Calculate gradients
            loss.backward()
            
            # Clip gradients if needed
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                
            # Update weights
            self.optimizer.step()
            
        # Update learning rate scheduler if batch-based
        if self.scheduler is not None and isinstance(self.scheduler, 
                                                    (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
            self.scheduler.step()
            
        # Calculate metrics
        metrics = {
            'loss': loss.item()
        }
        
        return metrics
        
    def predict_batch(self, batch: torch.Tensor) -> ModelOutput:
        """
        Make predictions on a single batch
        
        Args:
            batch: Batch of data
            
        Returns:
            Model predictions
        """
        # Move data to device
        batch = batch.to(self.config.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            output = self.model(batch)
            
        # Return predictions
        return ModelOutput(
            predictions=output.cpu().numpy(),
            probabilities=None,  # Not applicable for regression tasks
            raw_output=output
        )


def create_deep_learning_model(model_type: str, config: DeepLearningConfig, **kwargs: Any) -> DeepLearningModel:
    """Factory function to instantiate a deep learning model."""
    model_map = {
        "lstm": LSTMModel,
        "gru": GRUModel,
        "conv1d": Conv1DModel,
        "transformer": TransformerModel,
        "lstm_attention": LSTMAttentionModel,
        "wavenet": WaveNetModel,
        "inceptiontime": InceptionTimeModel,
        "hybrid": HybridModel,
    }
    cls = model_map.get(model_type.lower())
    if not cls:
        raise ModelError(f"Unknown deep learning model: {model_type}")
    return cls(config, **kwargs)

