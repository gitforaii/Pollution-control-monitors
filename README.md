# Pollution Control Monitors

Grid-based simulation where agents monitor air pollution, trigger alerts, and coordinate mitigation. The project records time-series metrics, visualizes them, and trains ML models (Random Forest / Gradient Boosting) on both simulation history and Kaggle CSV datasets.

## Features

- **Grid simulation**: 2D pollution field updated over discrete time steps.
- **Pollution sources**: Fixed emitters injecting pollution into the grid.
- **Monitoring agents**: Move on the grid, raise alerts, and mitigate high-pollution cells.
- **Metrics tracking**: Total, mean, and max pollution, number of alerts, and mitigation amount.
- **JSON export**: Simulation history saved for visualization and ML training.
- **Visualization**: Time-series plots, alerts vs mitigation, and pollution distribution.
- **ML training**: Random Forest and Gradient Boosting models on simulation JSON or Kaggle CSV data.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the simulation**
   ```bash
   python main.py            # writes simulation_history.json
   # or specify a custom output
   python main.py data/run1.json
   ```
3. **Visualize results**
   ```bash
   python visualize.py               # reads simulation_history.json
   python visualize.py data/run1.json
   ```
   You will see sequential popup windows. Close each plot window to see the next.

4. **Train on simulation history** (JSON produced by `main.py`)
   ```bash
   # Inside a Python shell or script
   from train_model import train_from_history
   model, metrics, features = train_from_history("simulation_history.json", model_type="rf")
   ```

5. **Train directly from a Kaggle CSV**
   ```bash
   python train_model.py data/air_quality.csv rf
   python train_model.py data/air_quality.csv gb
   ```

## Installation

1. Create a virtual environment (recommended).
2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How It Works

- `main.py` runs a grid simulation.
  - Pollution sources emit into fixed grid cells.
  - Monitoring agents perform a random walk, check local pollution, raise alerts, and mitigate if above a threshold.
  - At each step the simulation records:
    - `total_pollution`, `mean_pollution`, `max_pollution`, `alerts`, `mitigated_amount`.
  - Results are stored in a JSON history file.
- `visualize.py` reads the history JSON and creates:
  - Time-series of pollution levels over time.
  - Alerts vs mitigation activity.
  - Distribution of total pollution.
- `train_model.py` loads either JSON history or Kaggle CSV data and trains ML models.

## JSON Format (Simulation and Converted Kaggle Data)

Example JSON file produced by `main.py` or `convert_kaggle_csv_to_json`:

```json
{
  "config": {
    "width": 10,
    "height": 10,
    "steps": 200,
    "num_sources": 4,
    "num_agents": 6
  },
  "history": [
    {
      "step": 0,
      "total_pollution": 120.5,
      "mean_pollution": 3.2,
      "max_pollution": 9.8,
      "alerts": 2.0,
      "mitigated_amount": 15.0
    }
  ]
}
```

Required fields inside each `history` item:

- `step`: Integer time step index.
- `total_pollution`: Sum of pollution over the grid.
- `mean_pollution`: Average pollution per grid cell.
- `max_pollution`: Maximum pollution in any single cell.
- `alerts`: Number of agent alerts fired in that step.
- `mitigated_amount`: Total pollution reduced by mitigation actions.

## Training with Kaggle Datasets

`train_model.py` can consume Kaggle CSV files directly. Internally it calls `convert_kaggle_csv_to_json` to map numeric columns into the JSON history format above.

### Recommended Kaggle Datasets

Search these datasets on Kaggle (links point to Kaggle search pages):

1. **Beijing Multi-Site Air-Quality Data**  
   https://www.kaggle.com/datasets?search=Beijing%20Multi-Site%20Air-Quality%20Data
2. **Air Quality UCI Dataset**  
   https://www.kaggle.com/datasets?search=Air%20Quality%20UCI
3. **Delhi / India Air Quality Data**  
   https://www.kaggle.com/datasets?search=Delhi%20Air%20Quality
4. **Real-time Air Quality Index (AQI) Data**  
   https://www.kaggle.com/datasets?search=Air%20Quality%20Index

Any CSV with numeric pollution/air-quality measurements (e.g., PM2.5, PM10, NO2, AQI) can be used.

### Download via Kaggle CLI

1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```
2. Configure your API token following the Kaggle documentation.
3. Download a dataset, for example:
   ```bash
   kaggle datasets download -d <owner>/<dataset-name> -p data --unzip
   ```

### CSV Format Expectations

- The script looks for numeric columns in the CSV.
- The first numeric column is treated as `total_pollution`.
- The second numeric column (if present) is treated as `mean_pollution`.
- The third numeric column (if present) is treated as `max_pollution`.
- `alerts` and `mitigated_amount` are filled with zeros because they are usually not present in real datasets.

Resulting JSON entries look like:

```json
{
  "step": 0,
  "total_pollution": 35.1,
  "mean_pollution": 27.4,
  "max_pollution": 80.2,
  "alerts": 0.0,
  "mitigated_amount": 0.0
}
```

### Training Commands

```bash
# Random Forest on a Kaggle CSV
python train_model.py data/air_quality.csv rf

# Gradient Boosting on the same CSV
python train_model.py data/air_quality.csv gb
```

The script will:

- Convert the CSV to `<name>_converted.json`.
- Train the selected model.
- Print evaluation metrics (MSE, MAE, R²).
- Save the trained model as `<name>_<model_type>_model.pkl`.

## Training with Custom Datasets

You can build your own JSON history files that follow the same structure.

### JSON Requirements

- Top-level keys: `config` (optional) and `history` (required).
- `history` must be a list of step dictionaries with all required fields.
- Minimum recommended size: at least **100 steps** for meaningful training.

### Example Custom JSON

```json
{
  "config": {"width": 20, "height": 20, "steps": 500},
  "history": [
    {
      "step": 0,
      "total_pollution": 10.0,
      "mean_pollution": 0.5,
      "max_pollution": 1.5,
      "alerts": 0.0,
      "mitigated_amount": 0.0
    }
  ]
}
```

### Training from Custom JSON

```python
from train_model import train_from_history

model, metrics, feature_names = train_from_history("data/custom_history.json", model_type="gb")
print(metrics)
```

### Dataset Quality Guidelines

- Include realistic variation in pollution levels over time.
- Avoid constant or nearly constant series.
- Ensure numeric types (floats or integers) for all metric fields.
- More history steps generally produce better models.

### Customization Options

- Change grid size, number of sources, number of agents, and steps in `main.py`.
- Adjust agent thresholds and mitigation strength for different policies.
- Add new metrics to the JSON and extend `prepare_features` in `train_model.py` to use them.

## Project Usage Examples

- Run several simulations with different parameters and compare their pollution curves.
- Train a model on one run and use it to predict how policy changes might affect future steps.
- Use Kaggle city-level air-quality data to learn how pollution evolves day-to-day and compare it with the simulated grid dynamics.
