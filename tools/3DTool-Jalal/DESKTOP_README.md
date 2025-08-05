# 3D Cell Analysis Tool - Desktop Application

This is the original PyQt5-based desktop application for 3D cell analysis and visualization.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- PyQt5 (for GUI)
- Required packages (see requirements below)

### Installation

1. **Install Python Dependencies**
   ```bash
   pip install PyQt5 opencv-python numpy tifffile vtk scikit-learn matplotlib scipy
   ```

2. **Run the Application**
   ```bash
   python 3D_CellAnalysis_23.py
   ```

## 📋 Features

- **3D Image Processing**: Load and process multi-dimensional image stacks (3D, 4D, time-series)
- **Interactive ROI Selection**: Freeform drawing for region-of-interest selection and deletion
- **Advanced Segmentation**: Multiple methods including threshold, Otsu, adaptive, K-means, watershed, and ML Random Forest
- **Background Correction**: Advanced preprocessing with median and Gaussian filtering
- **Image Adjustments**: Real-time brightness/contrast controls with auto-adjustment
- **3D Surface Rendering**: VTK-based 3D visualization with marching cubes and volume rendering
- **Surface Management**: Filter, delete, and clean 3D surfaces with advanced options
- **Volume Analysis**: Comprehensive quantitative analysis with object metrics
- **Export Capabilities**: Save processed images, surfaces, analysis results, and logs
- **GPU Acceleration**: Optional GPU acceleration for faster processing
- **Multi-format Support**: TIFF, TIF, PNG, JPG, JPEG with TIFF recommended

## 📊 Analysis Capabilities

- **Multi-dimensional Support**: Handle 3D, 4D, and time-series data
- **Background Correction**: Advanced preprocessing algorithms
- **Image Enhancement**: Contrast adjustment and noise reduction
- **ROI Management**: Interactive region selection and manipulation
- **Pixel Classification**: Random Forest-based segmentation
- **Surface Rendering**: Generate 3D surfaces from segmented data
- **Volume Rendering**: Interactive 3D volume visualization

## 📁 Supported Data Formats

- **Input Formats**: TIFF, TIF, PNG, JPG, JPEG
- **3D Data**: Multi-slice images, confocal microscopy data
- **Time Series**: 4D data with temporal information
- **Recommended**: TIFF format for best compatibility

## 🔧 Troubleshooting

### Common Issues

1. **PyQt5 Installation Issues**
   - On Windows: `pip install PyQt5`
   - On Linux: `sudo apt-get install python3-pyqt5` (Ubuntu/Debian)
   - On macOS: `brew install pyqt5` (with Homebrew)

2. **OpenCV Issues**
   - Try: `pip install opencv-python` instead of `opencv-python-headless`
   - For headless servers: `pip install opencv-python-headless`

3. **VTK Installation Issues**
   - On Windows: `pip install vtk`
   - On Linux: May need system packages: `sudo apt-get install python3-vtk7`

4. **GUI Not Opening**
   - Ensure you have a display (for remote servers, use X11 forwarding)
   - On Windows, make sure you're running from a terminal with GUI support

5. **Memory Issues with Large Files**
   - Close other applications to free up RAM
   - Consider processing smaller chunks of data

## 📞 Support

For technical support or questions:
- **Email**: jalal.rahbani@hotmail.com
- **GitHub**: [@jalalrahbani](https://github.com/jalalrahbani)
- **Website**: [Personal Website](https://jalalrahbani.github.io/-jalal-website/)

## 🔄 Web Version

Prefer a web-based version? Try the Streamlit app:
- **Live Demo**: [3D Cell Analysis Web App](https://3d-tool-jalal-jalalrahbani.streamlit.app)

## 🛠️ System Requirements

- **RAM**: 8GB+ recommended for large 3D datasets
- **Storage**: Sufficient space for your image files
- **Display**: 1920x1080 or higher recommended for 3D visualization
- **GPU**: Optional but recommended for faster 3D rendering

---

*Developed with ❤️ for the research community* 