# F1 Fantasy Optimizer Web Application

A web-based tool for optimizing F1 Fantasy team selections using Value For Money (VFM) calculations and track affinity analysis.

## Features

- **VFM Calculation**: Analyzes driver and constructor performance with outlier removal
- **Track Affinity Analysis**: Determines how well drivers/constructors perform at specific track types
- **Two-Step Optimization**: Optimizes team selection for the next two races
- **Risk Tolerance Settings**: Choose between consistent performers or track-specific optimization
- **Interactive Web Interface**: Easy-to-use web UI for configuration and results
- **Default Data Support**: Upload data once and reuse it for multiple optimizations
- **Configuration Memory**: Automatically remembers your last team configuration
- **Docker Support**: Fully containerized for easy deployment
- **Optimized Performance**: Vectorized calculations and configurable bootstrapping speed up analyses by roughly 30%

## Prerequisites

- Docker and Docker Compose installed
- CSV data files:
  - `driver_race_data.csv`: Driver performance data
  - `constructor_race_data.csv`: Constructor performance data
  - `calendar.csv`: Race calendar with circuits
  - `tracks.csv`: Track characteristics

## Quick Start

1. **Clone or download the project files**

2. **Create the project structure**:
```bash
mkdir f1-optimizer
cd f1-optimizer

# Create directories
mkdir templates
mkdir default_data  # For storing default data files

# Save the provided files:
# - app.py (main Flask application)
# - f1_optimizer.py (optimization logic from the CLI version)
# - templates/index.html (web interface)
# - Dockerfile
# - docker-compose.yml
# - requirements.txt
```

3. **Prepare the f1_optimizer.py file**:
   
   Take the complete F1 optimizer code from the CLI version and save it as `f1_optimizer.py`, but remove the `main()` function and the `if __name__ == "__main__"` block at the end.

4. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

5. **Access the application**:
   
   Open your browser and go to `http://localhost:5000`

## Using the Application

### Initial Setup - Upload Default Data

1. **First Time Setup**:
   - Click on each file input to select your CSV files
   - Check "Save these files as default data"
   - Click "Upload New Files"
   - Your data is now saved and will persist between sessions

2. **Using Default Data**:
   - When you return to the app, click "Use Default Data"
   - Your previous configuration will be automatically loaded
   - You can immediately run optimizations without re-uploading files

### Running Optimizations

1. **Configure Your Team**:
   - Select your current 5 drivers and 2 constructors
   - Set your remaining budget
   - Choose swap limits for each step
   - Select weighting scheme and risk tolerance

2. **Run Optimization**:
   - Click "Run Optimization"
   - View recommended swaps for each race
   - See expected point improvements
   - Download full results as JSON

3. **Updating Data**:
   - To update default data, upload new files and check "Save these files as default data"
   - To use different data temporarily, upload files without checking the box

## Data File Formats

### driver_race_data.csv
```csv
Driver,Team,Cost,Race1,Race2,Race3,...
Max VERSTAPPEN,Red Bull,$30.5M,44,48,43,...
```

### constructor_race_data.csv
```csv
Constructor,Cost,Race1,Race2,Race3,...
Red Bull,$25.4M,73,78,73,...
```

### calendar.csv
```csv
Race,Grand Prix,Circuit
Race1,Bahrain Grand Prix,Bahrain International Circuit
Race2,Saudi Arabian Grand Prix,Jeddah Corniche Circuit
```

### tracks.csv
```csv
Grand Prix,Circuit,Corners,Length (km),Overtaking Opportunities,Track Speed,Expected Temperatures
Bahrain Grand Prix,Bahrain International Circuit,15,5.412,High,Medium,Hot
```

## Configuration Options

- **Weighting Scheme**: How to weight historical performance
  - Trend-based: Adaptive weights based on performance trends
  - Equal: All races weighted equally
  - Linear/Exponential decay: Recent races weighted more

- **Risk Tolerance**:
  - Low: Prioritizes consistent performers
  - Medium: Balanced approach
  - High: Prioritizes track-specific performance
  - **Bootstrap Iterations**: `bootstrap_iterations` controls the number of samples used when estimating track affinities (default 30)

## Data Persistence

The application supports three types of data persistence:

1. **Default Data**: CSV files stored in `default_data/` directory
   - Persists across Docker container restarts
   - Shared across all optimization sessions
   - Updated only when explicitly requested

2. **Configuration Memory**: Last used team configuration
   - Automatically saved after each optimization
   - Loaded when using default data

3. **Results History**: All optimization results
   - Saved in `results/` directory
   - Each result has a unique timestamp
   - Can be downloaded as JSON

## Development

### Running without Docker

1. Install Python 3.9+
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

### File Structure
```
f1-optimizer/
├── app.py                 # Flask web application
├── f1_optimizer.py        # Core optimization logic
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary uploaded files
├── results/              # Optimization results
├── default_data/         # Default CSV files (persisted)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Environment Variables

- `FLASK_SECRET_KEY`: Secret key for Flask sessions (default: auto-generated)
- `FLASK_ENV`: Flask environment (production/development)

## Troubleshooting

- **Port already in use**: Change the port in docker-compose.yml and app.py
- **File upload errors**: Ensure CSV files are properly formatted
- **Default data not loading**: Check that files exist in `default_data/` directory
- **Optimization errors**: Check that driver/constructor names match exactly between files
- **Memory issues**: For large datasets, increase Docker memory allocation

## Security Notes

- This is designed for local/private use
- No authentication is implemented
- Uploaded files are stored locally
- Consider adding authentication for public deployment

## License

This project is provided as-is for educational and personal use.1,Race2,Race3,...
Max VERSTAPPEN,Red Bull,$30.5M,44,48,43,...
```

### constructor_race_data.csv
```csv
Constructor,Cost,Race1,Race2,Race3,...
Red Bull,$25.4M,73,78,73,...
```

### calendar.csv
```csv
Race,Grand Prix,Circuit
Race1,Bahrain Grand Prix,Bahrain International Circuit
Race2,Saudi Arabian Grand Prix,Jeddah Corniche Circuit
```

### tracks.csv
```csv
Grand Prix,Circuit,Corners,Length (km),Overtaking Opportunities,Track Speed,Expected Temperatures
Bahrain Grand Prix,Bahrain International Circuit,15,5.412,High,Medium,Hot
```

## Configuration Options

- **Weighting Scheme**: How to weight historical performance
  - Trend-based: Adaptive weights based on performance trends
  - Equal: All races weighted equally
  - Linear/Exponential decay: Recent races weighted more

- **Risk Tolerance**:
  - Low: Prioritizes consistent performers
  - Medium: Balanced approach
  - High: Prioritizes track-specific performance
  - **Bootstrap Iterations**: `bootstrap_iterations` controls the number of samples used when estimating track affinities (default 30)

## Development

### Running without Docker

1. Install Python 3.9+
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

### File Structure
```
f1-optimizer/
├── app.py                 # Flask web application
├── f1_optimizer.py        # Core optimization logic
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Uploaded CSV files (created automatically)
├── results/              # Optimization results (created automatically)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Troubleshooting

- **Port already in use**: Change the port in docker-compose.yml and app.py
- **File upload errors**: Ensure CSV files are properly formatted
- **Optimization errors**: Check that driver/constructor names match exactly between files
- **Memory issues**: For large datasets, increase Docker memory allocation

## Security Notes

- This is designed for local/private use
- No authentication is implemented
- Uploaded files are stored locally
- Consider adding authentication for public deployment

## License

This project is provided as-is for educational and personal use.
