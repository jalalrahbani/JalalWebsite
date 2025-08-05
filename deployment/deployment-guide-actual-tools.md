# Deployment Guide for Your Actual Tools

This guide covers how to deploy your actual Python applications (CellTrack Official and 3D Cell Analysis Tool) to GitHub and Streamlit Cloud.

## 🚀 Quick Deployment Steps

### Step 1: Create GitHub Repositories

1. **CellTrack Official Repository**
   - Go to [github.com](https://github.com)
   - Create new repository: `celltrack-official`
   - Make it public
   - Upload files from `tools/CellTrack official/` folder

2. **3D Cell Analysis Tool Repository**
   - Create new repository: `3d-tool-jalal`
   - Make it public
   - Upload files from `tools/3DTool-Jalal/` folder

### Step 2: Deploy to Streamlit Cloud

1. **Deploy CellTrack Official**
   - Go to [streamlit.io](https://streamlit.io)
   - Sign up with GitHub
   - Click "New app"
   - Repository: `celltrack-official`
   - File path: `celltrack_app.py`
   - Click "Deploy"

2. **Deploy 3D Cell Analysis Tool**
   - Click "New app" again
   - Repository: `3d-tool-jalal`
   - File path: `3d_tool_app.py`
   - Click "Deploy"

### Step 3: Update Website Links

After deployment, update these files with the live URLs:

## 📁 File Structure for Each Repository

### CellTrack Official Repository
```
celltrack-official/
├── celltrack_app.py          # Main Streamlit application
├── requirements.txt           # Python dependencies
├── packages.txt              # System dependencies (if needed)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # Documentation
```

### 3D Cell Analysis Tool Repository
```
3d-tool-jalal/
├── 3d_tool_app.py           # Main Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies
├── .streamlit/
│   └── config.toml         # Streamlit configuration
└── README.md               # Documentation
```

## 🔧 Configuration Files

### requirements.txt (CellTrack Official)
```txt
pandas>=2.1.0
numpy>=1.26.0
matplotlib>=3.7.0
scipy>=1.11.0
seaborn>=0.12.0
openpyxl>=3.1.0
streamlit>=1.28.0
plotly>=5.15.0
Pillow>=10.0.0
```

### requirements.txt (3D Cell Analysis Tool)
```txt
opencv-python-headless>=4.8.0
numpy>=1.26.0
tifffile>=2023.0.0
vtk>=9.2.0
scikit-learn>=1.3.0
streamlit>=1.28.0
plotly>=5.15.0
Pillow>=10.0.0
matplotlib>=3.7.0
scipy>=1.11.0
```

### packages.txt (3D Tool only)
```txt
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
libvtk9-dev
libvtk9-qt-dev
```

### .streamlit/config.toml (Both tools)
```toml
[server]
headless = true
enableCORS = false
port = 8501

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🔗 Update Website Links

After deploying the tools, update these files with the live URLs:

### 1. Update projects.html
```html
<!-- CellTrack Official -->
<a href="https://celltrack-official-jalalrahbani.streamlit.app" target="_blank">Launch Interactive Tool</a>

<!-- 3D Cell Analysis Tool -->
<a href="https://3d-tool-jalal-jalalrahbani.streamlit.app" target="_blank">Launch Interactive Tool</a>
```

### 2. Update tool HTML pages
```html
<!-- tools/CellTrack official/celltrack-tool.html -->
<a href="https://celltrack-official-jalalrahbani.streamlit.app" target="_blank" class="launch-button">
    🚀 Launch CellTrack Official
</a>

<!-- tools/3DTool-Jalal/3d-tool.html -->
<a href="https://3d-tool-jalal-jalalrahbani.streamlit.app" target="_blank" class="launch-button">
    🚀 Launch 3D Cell Analysis Tool
</a>
```

### 3. Add iframe embeds (optional)
```html
<!-- For CellTrack Official -->
<iframe 
    src="https://celltrack-official-jalalrahbani.streamlit.app" 
    class="tool-iframe"
    title="CellTrack Official - Cell Migration Analysis Tool">
</iframe>

<!-- For 3D Cell Analysis Tool -->
<iframe 
    src="https://3d-tool-jalal-jalalrahbani.streamlit.app" 
    class="tool-iframe"
    title="3D Cell Analysis Tool">
</iframe>
```

## 🐛 Troubleshooting

### Common Issues

1. **Python Version Compatibility**
   - Use `requirements.txt` with `>=` instead of `==`
   - Ensure packages are compatible with Python 3.13

2. **OpenCV Issues**
   - Use `opencv-python-headless` instead of `opencv-python`
   - Include `packages.txt` for system dependencies

3. **VTK Issues (3D Tool)**
   - May require additional system dependencies
   - Consider using a simpler version without VTK for web deployment

4. **Streamlit Deployment Failures**
   - Check that all files are in the correct repository
   - Ensure `requirements.txt` is in the root directory
   - Verify the app file path is correct

5. **Website Not Updating**
   - Wait 5-10 minutes for GitHub Pages to rebuild
   - Check the Actions tab for deployment status
   - Clear browser cache

## 📞 Support

For deployment issues:
- **Email**: jalal.rahbani@hotmail.com
- **GitHub**: [@jalalrahbani](https://github.com/jalalrahbani)
- **Website**: [Personal Website](https://jalalrahbani.github.io/-jalal-website/)

## 🎯 Next Steps

1. **Create GitHub repositories** for both tools
2. **Upload the deployment files** to each repository
3. **Deploy to Streamlit Cloud** using the provided configuration
4. **Update website links** with the live URLs
5. **Test the tools** to ensure they work correctly
6. **Share your work** with the research community!

---

*Last updated: August 2025* 