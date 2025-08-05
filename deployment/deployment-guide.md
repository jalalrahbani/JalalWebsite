# Deployment Guide

This guide covers how to deploy the main website and interactive tools.

## 🌐 Main Website Deployment

### GitHub Pages Deployment

1. **Create GitHub Repository**
   - Go to [github.com](https://github.com)
   - Create new repository: `jalal-website`
   - Make it public

2. **Upload Website Files**
   - Upload all HTML files to the root directory
   - Upload `assets/` folder with CSS and images
   - Upload `README.md`

3. **Enable GitHub Pages**
   - Go to repository Settings
   - Scroll to "Pages" section
   - Source: "Deploy from a branch"
   - Branch: `main`
   - Folder: `/ (root)`
   - Click "Save"

4. **Access Your Website**
   - URL: `https://yourusername.github.io/jalal-website/`
   - Updates automatically when you push changes

## 🚀 Interactive Tools Deployment

### CellTrack Official

1. **Create GitHub Repository**
   - Repository name: `CellTrack-official`
   - Make it public

2. **Upload Files**
   ```
   CellTrack-official/
   ├── celltrack_app.py
   ├── requirements.txt
   ├── .streamlit/
   │   └── config.toml
   └── README.md
   ```

3. **Deploy to Streamlit Cloud**
   - Go to [streamlit.io](https://streamlit.io)
   - Sign up with GitHub
   - Click "New app"
   - Repository: `CellTrack-official`
   - File path: `celltrack_app.py`
   - Click "Deploy"

### 3D Cell Analysis Tool

1. **Create GitHub Repository**
   - Repository name: `3DTool-Jalal`
   - Make it public

2. **Upload Files**
   ```
   3DTool-Jalal/
   ├── 3d_tool_app.py
   ├── requirements.txt
   ├── packages.txt
   ├── .streamlit/
   │   └── config.toml
   └── README.md
   ```

3. **Deploy to Streamlit Cloud**
   - Click "New app" again
   - Repository: `3DTool-Jalal`
   - File path: `3d_tool_app.py`
   - Click "Deploy"

## 🔧 Configuration Files

### requirements.txt
```txt
streamlit>=1.28.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.15.0
Pillow>=10.0.0
opencv-python-headless>=4.8.0
scikit-image>=0.21.0
matplotlib>=3.7.0
```

### packages.txt
```txt
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
```

### .streamlit/config.toml
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

### projects.html
```html
<a href="https://celltrack-official-jalalrahbani.streamlit.app" target="_blank">Launch Interactive Tool</a>
<a href="https://3d-tool-jalal-jalalrahbani.streamlit.app" target="_blank">Launch Interactive Tool</a>
```

### celltrack-tool.html
```html
<iframe 
    src="https://celltrack-official-jalalrahbani.streamlit.app" 
    class="tool-iframe"
    title="CellTrack Official">
</iframe>
```

### 3d-tool.html
```html
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

3. **Streamlit Deployment Failures**
   - Check that all files are in the correct repository
   - Ensure `requirements.txt` is in the root directory
   - Verify the app file path is correct

4. **Website Not Updating**
   - Wait 5-10 minutes for GitHub Pages to rebuild
   - Check the Actions tab for deployment status
   - Clear browser cache

## 📞 Support

For deployment issues:
- **Email**: jalal.rahbani@hotmail.com
- **GitHub**: [@jalalrahbani](https://github.com/jalalrahbani)
- **Website**: [Personal Website](https://jalalrahbani.github.io/-jalal-website/)

---

*Last updated: August 2025* 