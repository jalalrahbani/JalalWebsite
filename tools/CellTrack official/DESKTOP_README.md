# CellTrack Official - Desktop Application

This is the original Tkinter-based desktop application for cell migration analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements below)

### Installation

1. **Install Python Dependencies**
   ```bash
   pip install pandas numpy matplotlib scipy seaborn openpyxl
   ```

2. **Run the Application**
   ```bash
   python final_cell_tracking_analysis_GUI_2.py
   ```

## 📋 Features

- **Excel File Processing**: Handle multiple sheets with different experimental conditions
- **Advanced Cell Tracking Analysis**: Comprehensive metrics including speed, directionality, persistence, MSD, and more
- **Statistical Comparisons**: T-tests, Mann-Whitney U tests, and detailed condition comparisons
- **Interactive Visualizations**: Multiple plot types with publication-ready figures
- **Export Results**: Color-coded Excel output with comprehensive analysis reports
- **Multi-track Analysis**: Process multiple cell tracks simultaneously with individual track analysis
- **Real-time Analysis**: Live processing with immediate visualization updates
- **Data Validation**: Automatic validation of required columns and data format
- **Advanced Metrics**: Directionality over time/distance, path tortuosity, straightness index
- **Statistical Testing**: Comprehensive statistical analysis with significance testing

## 📊 Analysis Metrics

- **Total Distance**: Cumulative distance traveled by cells
- **Net Displacement**: Straight-line distance from start to end
- **Average Speed**: Mean velocity of cell movement
- **Directionality Ratio**: Net displacement / total distance
- **Persistence Time**: Time spent moving in consistent direction
- **Mean Squared Displacement (MSD)**: Measure of cell motility

## 📁 Required Data Format

Your Excel file should contain these columns:
- **track number**: Cell identifier
- **slice number**: Time point
- **X coordinate**: X position
- **Y coordinate**: Y position
- **time (min)**: Time in minutes

Multiple sheets can represent different experimental conditions.

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   - Make sure all required packages are installed
   - Try: `pip install --upgrade pandas numpy matplotlib scipy seaborn openpyxl`

2. **GUI Not Opening**
   - Ensure you have a display (for remote servers, use X11 forwarding)
   - On Windows, make sure you're running from a terminal with GUI support

3. **File Loading Issues**
   - Check that your Excel file is in the correct format
   - Ensure the file is not corrupted or password-protected

## 📞 Support

For technical support or questions:
- **Email**: jalal.rahbani@hotmail.com
- **GitHub**: [@jalalrahbani](https://github.com/jalalrahbani)
- **Website**: [Personal Website](https://jalalrahbani.github.io/-jalal-website/)

## 🔄 Web Version

Prefer a web-based version? Try the Streamlit app:
- **Live Demo**: [CellTrack Official Web App](https://celltrack-official-jalalrahbani.streamlit.app)

---

*Developed with ❤️ for the research community* 