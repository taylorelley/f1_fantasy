# F1 Fantasy Optimizer Web Application

A web-based tool for optimizing F1 Fantasy team selections using Value For Money (VFM) calculations and track affinity analysis.

## Features

- **VFM Calculation**: Analyzes driver and constructor performance with outlier removal
- **Track Affinity Analysis**: Determines how well drivers/constructors perform at specific track types
- **Two-Step Optimization**: Optimizes team selection for the next two races
- **Risk Tolerance Settings**: Choose between consistent performers or track-specific optimization
- **Interactive Web Interface**: Easy-to-use web UI for configuration and results
- **Docker Support**: Fully containerized for easy deployment

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

### Step 1: Upload Data Files
- Click on each file input to select your CSV files
- All four files are required
- Click "Upload Files" to process

### Step 2: Configure Optimization
- Select your current team (5 drivers, 2 constructors)
- Set your remaining budget
- Choose swap limits for each step
- Select weighting scheme and risk tolerance
- Click "Run Optimization"

### Step 3: View Results
- See recommended swaps for each race
- View expected point improvements
- Check final team composition
- Download full results as JSON

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
