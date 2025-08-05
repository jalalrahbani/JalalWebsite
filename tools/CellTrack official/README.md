# CellTrack Official - Advanced Cell Migration Analysis Tool

A comprehensive cell migration analysis tool for processing and analyzing cell tracking data from time-lapse microscopy experiments.

## 🚀 Live Demo

[Streamlit App Link - Coming Soon]

## 📋 Features

- **Excel File Processing**: Handle multiple sheets with different experimental conditions
- **Advanced Cell Tracking Analysis**: Calculate speed, directionality, persistence, and more
- **Statistical Comparisons**: Compare different experimental conditions
- **Interactive Visualizations**: Plot metrics and generate publication-ready figures
- **Export Results**: Save analysis results with color-coded Excel output
- **Multi-track Analysis**: Process multiple cell tracks simultaneously

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **SciPy**: Statistical analysis
- **OpenPyXL**: Excel file handling

## 📊 Analysis Metrics

### Cell Movement Analysis
- **Total Distance**: Cumulative distance traveled by cells
- **Net Displacement**: Straight-line distance from start to end
- **Average Speed**: Mean velocity of cell movement
- **Instantaneous Speed**: Speed at each time point
- **Directionality Ratio**: Net displacement / total distance
- **Persistence Time**: Time spent moving in consistent direction
- **Path Tortuosity**: Total distance / net displacement

### Statistical Analysis
- **Mean Squared Displacement (MSD)**: Measure of cell motility
- **Turning Angles**: Directional changes over time
- **Directionality Over Time**: How directionality changes during migration
- **Directionality Over Distance**: Directionality at different distance thresholds

## 📈 Supported Data Formats

- **Input**: Excel files (.xlsx) with cell tracking data
- **Required Columns**: 
  - `track number`: Cell identifier
  - `slice number`: Time point
  - `X coordinate`: X position
  - `Y coordinate`: Y position
  - `time (min)`: Time in minutes
- **Output**: Excel files with analysis results and visualizations

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jalalrahbani/celltrack-official.git
   cd celltrack-official
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run celltrack_app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

### Using the Web App

1. **Upload Data**: Use the file uploader to select your Excel file
2. **Set Parameters**: Adjust pixel size and analysis options
3. **Run Analysis**: Click "Analyze" to process your data
4. **View Results**: Explore interactive visualizations and statistics
5. **Download Results**: Export analysis results and plots

## 📊 Analysis Workflow

### Step 1: Data Upload
- Upload Excel file with cell tracking data
- Verify data format and required columns
- Set pixel size for accurate measurements

### Step 2: Analysis
- Process each cell track individually
- Calculate movement metrics
- Generate statistical comparisons

### Step 3: Visualization
- Create interactive plots
- Compare different conditions
- Generate publication-ready figures

### Step 4: Export
- Download analysis results
- Save visualizations
- Export statistical summaries

## 🔧 Parameters

### Analysis Settings
- **Pixel Size**: Physical size of pixels (μm/pixel)
- **Time Interval**: Time between frames (minutes)
- **Minimum Track Length**: Minimum number of points per track
- **Speed Threshold**: Minimum speed for analysis

### Visualization Options
- **Plot Type**: Speed, directionality, MSD, etc.
- **Color Scheme**: Different colors for conditions
- **Export Format**: PNG, PDF, SVG

## 📝 Usage Examples

### Example 1: Single Condition Analysis
1. Upload Excel file with one sheet
2. Set pixel size to 0.5 μm/pixel
3. Run analysis to get cell movement metrics
4. View speed and directionality plots

### Example 2: Multi-condition Comparison
1. Upload Excel file with multiple sheets
2. Each sheet represents different experimental conditions
3. Compare cell behavior between conditions
4. Generate statistical comparisons

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For technical support or questions:
- **Email**: jalal.rahbani@hotmail.com
- **Website**: [Personal Website](https://jalalrahbani.github.io/-jalal-website/)
- **GitHub**: [@jalalrahbani](https://github.com/jalalrahbani)

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Developer

**Jalal Al Rahbani**
- Research Scientist, BioData Analyst, Python Programmer & Automator
- McGill University, Department of Physiology
- Specializing in advanced light microscopy and image analysis

---

*Developed with ❤️ for the research community* 