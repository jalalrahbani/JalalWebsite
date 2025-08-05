# 3D Cell Analysis Tool - Advanced 3D Image Processing

A sophisticated 3D cell analysis and visualization tool for processing biological volumes, performing machine learning segmentation, and creating interactive 3D visualizations.

## 🚀 Live Demo

https://3dtool-jalal.streamlit.app

## 📋 Features

- **3D Image Processing**: Load and process multi-dimensional image stacks
- **ROI Selection**: Interactive region-of-interest selection with freeform drawing
- **Machine Learning Segmentation**: Pixel classification using Random Forest
- **3D Surface Rendering**: VTK-based 3D visualization and surface generation
- **Background Correction**: Advanced image preprocessing and enhancement
- **Volume Analysis**: Quantitative analysis of 3D structures
- **Export Capabilities**: Save processed images, surfaces, and analysis results

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **VTK**: 3D visualization and surface rendering
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **tifffile**: TIFF image format support
- **Plotly**: Interactive visualizations

## 📊 Analysis Capabilities

### Image Processing
- **Multi-dimensional Support**: Handle 3D, 4D, and time-series data
- **Background Correction**: Advanced preprocessing algorithms
- **Image Enhancement**: Contrast adjustment and noise reduction
- **ROI Management**: Interactive region selection and manipulation

### Machine Learning
- **Pixel Classification**: Random Forest-based segmentation
- **Training Interface**: Interactive annotation and model training
- **Batch Processing**: Process entire image stacks automatically
- **Model Export**: Save trained models for reuse

### 3D Visualization
- **Surface Rendering**: Generate 3D surfaces from segmented data
- **Volume Rendering**: Interactive 3D volume visualization
- **Surface Filtering**: Remove noise and artifacts
- **Export Formats**: Save as OBJ, STL, or other 3D formats

## 📈 Supported Data Formats

- **Input**: TIFF, TIF, PNG, JPG, JPEG image stacks
- **3D Data**: Multi-slice images, confocal microscopy data
- **Time Series**: 4D data with temporal information
- **Output**: Processed images, 3D surfaces, analysis results

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jalalrahbani/3d-tool-jalal.git
   cd 3d-tool-jalal
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run 3d_tool_app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

### Using the Web App

1. **Upload Data**: Use the file uploader to select your 3D image stack
2. **Preprocess**: Apply background correction and image enhancement
3. **Segment**: Use ML tools or manual ROI selection
4. **Visualize**: Generate 3D surfaces and interactive visualizations
5. **Export**: Download processed data and analysis results

## 📊 Analysis Workflow

### Step 1: Data Loading
- Upload 3D image stack (TIFF format recommended)
- Verify data dimensions and format
- Apply initial preprocessing

### Step 2: Image Processing
- Background correction and noise reduction
- Contrast enhancement and normalization
- ROI selection and management

### Step 3: Segmentation
- Manual ROI selection with freeform drawing
- Machine learning pixel classification
- Batch processing of entire stacks

### Step 4: 3D Visualization
- Generate 3D surfaces from segmented data
- Interactive 3D rendering and exploration
- Surface filtering and cleaning

### Step 5: Analysis & Export
- Quantitative volume analysis
- Export processed images and surfaces
- Save analysis results and visualizations

## 🔧 Parameters

### Image Processing
- **Background Correction**: Algorithm selection and parameters
- **Contrast Enhancement**: Brightness and contrast adjustment
- **Noise Reduction**: Filtering and smoothing options

### Segmentation
- **ROI Selection**: Manual drawing and threshold-based selection
- **ML Parameters**: Random Forest classifier settings
- **Batch Processing**: Processing options for large datasets

### 3D Visualization
- **Surface Generation**: Marching cubes parameters
- **Rendering Options**: Lighting, materials, and colors
- **Export Settings**: Format and quality options

## 📝 Usage Examples

### Example 1: Basic 3D Analysis
1. Upload 3D TIFF stack
2. Apply background correction
3. Select ROIs manually
4. Generate 3D surfaces
5. Export results

### Example 2: ML Segmentation
1. Load image stack
2. Annotate training data
3. Train Random Forest classifier
4. Apply to entire stack
5. Generate 3D visualization

### Example 3: Time Series Analysis
1. Upload 4D time series data
2. Process each time point
3. Track changes over time
4. Create animated visualizations

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
