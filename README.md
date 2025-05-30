# QuantumSpectre Elite Trading System

A high-performance, production-grade trading system with advanced AI capabilities for cryptocurrency and forex markets. This system specializes in identifying market inefficiencies, loopholes, and exploiting patterns to achieve exceptionally high win rates.

## Key Features

- **Advanced Multi-Layer AI**: Combines multiple specialized intelligence systems for market analysis and prediction
- **Dynamic Brain Council**: Adaptive decision-making system that evolves based on performance metrics
- **Loophole Detection**: Specialized systems for identifying and exploiting market inefficiencies
- **Pattern Recognition**: Advanced detection of harmonic patterns, candlestick formations, and complex chart structures
- **Multi-Exchange Support**: Full integration with Binance and Deriv platforms
- **Real-Time Information Gathering**: Web crawlers, news analyzers, social media monitors, and darkweb intelligence
- **Sophisticated Risk Management**: Capital preservation with adaptive position sizing and circuit breakers
- **Continuous Evolution**: Self-learning systems that adapt and improve based on market conditions
- **Comprehensive Backtesting**: Rigorous historical testing and strategy validation
- **High-Performance Architecture**: Low-latency, scalable design optimized for real-time trading

## System Architecture

The QuantumSpectre Elite Trading System is built with a modular, microservice-based architecture that includes:

- **Core Infrastructure**: Configuration management, logging, and service orchestration
- **Data Ingestion**: Real-time market data collection from multiple exchanges
- **Feature Calculation**: Advanced technical indicator and pattern calculation
- **Intelligence System**: Multiple specialized trading brains and analysis systems
- **Brain Council**: Decision coordination and strategy weighting based on performance
- **Execution Engine**: Precise order execution with market microstructure awareness
- **Risk Management**: Sophisticated capital preservation and exposure control
- **Monitoring & Analytics**: Real-time performance tracking and system health monitoring
- **User Interface**: Comprehensive dashboard for monitoring and control

## Installation

### Prerequisites

 - Python 3.11+
- Redis
- PostgreSQL
- Node.js 16+ (for UI)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/quantum-spectre.git
   cd quantum-spectre
   ```

2. Install the base Python dependencies:
   ```bash
   pip install -r requirements-core.txt
   ```
   These include `numpy`, `pandas`, `PyYAML`, and `python-dateutil`, which are
   required for running the test suite. Optional packages can be added later
   with `pip install -r requirements.txt`.

   The optional ML stack now requires `scikit-learn>=1.3,<2.0`. Ensure
   your environment includes compatible versions of `shap`, `boruta`, and
   `category_encoders` if you install the full requirements file.
   
   The `prophet` package now requires `cmdstanpy` and a C++ toolchain for
   compilation. If installation fails with pip, use conda:
   ```bash
  conda install -c conda-forge prophet
  ```

If certain optional dependencies like TensorFlow or PyTorch are missing,
the system will automatically disable related features and continue to
operate with the lightweight components used during testing.

3. Run the environment setup script:
   ```bash
   bash scripts/setup_environment.sh
   ```
   This installs system packages, creates the RAPIDS-enabled conda environment, installs optional Python requirements from `requirements.txt`, downloads NLTK data, and installs the UI dependencies.

4. Create configuration:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize database:
   ```
   python -m scripts.init_db
   ```

6. Start the system:
   ```
   python main.py
   ```

7. In a separate terminal, start the ui for development:
   ```
   cd ui
   npm start
   ```

8. To create a production build of the ui:
   ```
   npm run build
   ```

9. Access the dashboard at http://localhost:8000

## Usage

### Configuration

The system behavior can be configured through environment variables or a configuration file. See `.env` for available options.

You can access a top-level section at runtime with `Config.get_section('backtester')`. If no file is loaded, values come from `DEFAULT_CONFIG`.

```python
from config import Config

api_cfg = Config.get_section('api')
```

### API Keys

To connect to exchanges, add your API keys to the `.env` file:

```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

DERIV_API_KEY=your_deriv_api_key
DERIV_API_SECRET=your_deriv_api_secret
```

### Trading Settings

Adjust risk parameters, strategy settings, and execution preferences in the dashboard UI or directly in the configuration.

### ML Model Types

Use `ml_models.type` to switch between supervised and reinforcement learning. Set it to `reinforcement` to
train a reinforcement learning agent.

## Advanced Features

### Custom Strategies

Create custom trading brains by extending the `BaseBrain` class:

```python
from strategy_brains.base_brain import BaseBrain

class MyCustomBrain(BaseBrain):
    def __init__(self, name="MyCustomBrain", config=None):
        super().__init__(name, config)
        
    def evaluate(self, features):
        # Implement your strategy logic here
        return {"signal": "BUY", "confidence": 0.85}
```

### Hardware Acceleration

Enable GPU acceleration for machine learning models by setting:

```
ML_HARDWARE_ACCELERATION=True
```

### Distributed Deployment

For high-performance deployments, use the Docker and Kubernetes configurations in the `deployment` directory.

## Performance Optimization

- Use uvloop on Unix-based systems for increased asyncio performance
- Enable hardware acceleration for machine learning components
- Adjust thread and worker counts based on your hardware
- Consider colocated deployment for reduced latency

## Security Considerations

- Store API keys securely using environment variables or a secrets manager
- Use read-only API keys when possible
- Enable two-factor authentication for exchange accounts
- Regularly rotate API keys
- Monitor for unauthorized access
- Ensure compliance with local laws when collecting data from online sources

### Dark Web Feed Legal Notice

The optional dark web intelligence feed is **disabled by default**. Gathering
information from the dark web can expose you to legal and ethical liabilities.
Only enable this feed if you understand the laws in your jurisdiction and have a
legitimate reason for monitoring public dark web content. The project
maintainers do not condone or support any illegal activity.

To enable the feed, set `data_feeds.dark_web.enabled` to `true` in your YAML
configuration. Leave it `false` (the default) to opt out.

```yaml
data_feeds:
  dark_web:
    enabled: true   # opt-in
    tor_proxy: "socks5h://localhost:9050"
```

Disabling is as simple as:

```yaml
data_feeds:
  dark_web:
    enabled: false  # default
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

Trading cryptocurrencies and forex involves significant risk. This software is for educational and research purposes only. Always test thoroughly using paper trading before deploying with real funds. Past performance is not indicative of future results.
