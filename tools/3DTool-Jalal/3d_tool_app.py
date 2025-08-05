import streamlit as st
import numpy as np
import cv2
import tifffile
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure

# Page configuration
st.set_page_config(
    page_title="3D Cell Analysis Tool - Enhanced",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSlider > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">3D Cell Analysis Tool - Enhanced</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced 3D Image Processing & Analysis with Full Feature Set</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'image_stack' not in st.session_state:
    st.session_state.image_stack = None
if 'corrected_stack' not in st.session_state:
    st.session_state.corrected_stack = None
if 'segmented_stack' not in st.session_state:
    st.session_state.segmented_stack = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'brightness' not in st.session_state:
    st.session_state.brightness = 0
if 'contrast' not in st.session_state:
    st.session_state.contrast = 100
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'show_corrected' not in st.session_state:
    st.session_state.show_corrected = False

# Helper functions
def load_image_stack(file):
    """Load image stack from uploaded file"""
    try:
        if file.name.lower().endswith(('.tif', '.tiff')):
            # Load TIFF stack
            image_stack = tifffile.imread(file)
        else:
            # Load single image
            image = Image.open(file)
            image_stack = np.array(image)
            if len(image_stack.shape) == 2:
                image_stack = image_stack[np.newaxis, :, :]
        
        return image_stack
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def apply_background_correction(image_stack):
    """Apply advanced background correction to image stack"""
    corrected_stack = np.zeros_like(image_stack, dtype=np.float32)
    
    for i in range(image_stack.shape[0]):
        # Get current slice
        slice_img = image_stack[i]
        
        # Calculate background using multiple methods
        # Method 1: Median filter
        background_median = cv2.medianBlur(slice_img.astype(np.uint8), 51)
        
        # Method 2: Gaussian blur for smoother background
        background_gaussian = cv2.GaussianBlur(slice_img.astype(np.uint8), (51, 51), 0)
        
        # Combine methods
        background = (background_median + background_gaussian) / 2
        
        # Subtract background
        corrected = slice_img.astype(np.float32) - background.astype(np.float32)
        corrected = np.clip(corrected, 0, 255)
        
        corrected_stack[i] = corrected
    
    return corrected_stack.astype(np.uint8)

def adjust_brightness_contrast(image, brightness=0, contrast=100):
    """Adjust brightness and contrast of image"""
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Apply brightness
    img_float = img_float + brightness
    
    # Apply contrast
    img_float = img_float * (contrast / 100.0)
    
    # Clip to valid range
    img_float = np.clip(img_float, 0, 255)
    
    return img_float.astype(np.uint8)

def segment_slice_advanced(slice_img, method='threshold', threshold_value=128, kmeans_clusters=3, watershed_markers=10):
    """Advanced segmentation with multiple methods"""
    if method == 'threshold':
        # Manual thresholding
        _, segmented = cv2.threshold(slice_img, threshold_value, 255, cv2.THRESH_BINARY)
    elif method == 'otsu':
        # Otsu thresholding
        _, segmented = cv2.threshold(slice_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # Adaptive thresholding
        segmented = cv2.adaptiveThreshold(slice_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'kmeans':
        # K-means clustering
        reshaped = slice_img.reshape(-1, 1)
        kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
        labels = kmeans.fit_predict(reshaped)
        segmented = labels.reshape(slice_img.shape) * (255 // (kmeans_clusters - 1))
    elif method == 'watershed':
        # Watershed segmentation
        _, thresh = cv2.threshold(slice_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        segmented = cv2.watershed(cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR), markers)
        segmented = (segmented > 1).astype(np.uint8) * 255
    elif method == 'ml_random_forest':
        # Machine learning segmentation using Random Forest
        # Create training data from image
        features = slice_img.reshape(-1, 1)
        
        # Simple threshold-based labeling for training
        _, labels = cv2.threshold(slice_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        labels = labels.reshape(-1)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        
        # Predict
        predictions = rf.predict(features)
        segmented = predictions.reshape(slice_img.shape)
    
    return segmented

def create_3d_surface_enhanced(segmented_stack, surface_method='marching_cubes'):
    """Create enhanced 3D surface from segmented stack"""
    if surface_method == 'marching_cubes':
        # Use marching cubes for surface generation
        from skimage import measure
        
        # Create 3D surface using marching cubes
        verts, faces, normals, values = measure.marching_cubes(segmented_stack, level=0.5)
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1], 
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.8,
                color='lightblue',
                flatshading=True
            )
        ])
        
        fig.update_layout(
            title="3D Surface Visualization",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z"
            ),
            width=800,
            height=600
        )
        
        return fig
    
    elif surface_method == 'volume_rendering':
        # Volume rendering approach
        fig = go.Figure(data=go.Volume(
            x=np.arange(segmented_stack.shape[2]),
            y=np.arange(segmented_stack.shape[1]),
            z=np.arange(segmented_stack.shape[0]),
            value=segmented_stack.flatten(),
            opacity=0.3,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="3D Volume Rendering",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            width=800,
            height=600
        )
        
        return fig

def calculate_volume_metrics_enhanced(segmented_stack, pixel_size=1.0):
    """Calculate comprehensive volume metrics"""
    # Basic metrics
    total_voxels = np.sum(segmented_stack > 0)
    volume = total_voxels * (pixel_size ** 3)
    
    # Surface area calculation
    from skimage import measure
    try:
        verts, faces, _, _ = measure.marching_cubes(segmented_stack, level=0.5)
        surface_area = measure.mesh_surface_area(verts, faces) * (pixel_size ** 2)
    except:
        surface_area = 0
    
    # Connectivity analysis
    labeled_stack, num_objects = ndimage.label(segmented_stack)
    
    # Calculate metrics for each object
    object_metrics = []
    for i in range(1, num_objects + 1):
        object_mask = (labeled_stack == i)
        object_voxels = np.sum(object_mask)
        object_volume = object_voxels * (pixel_size ** 3)
        
        # Calculate centroid
        coords = np.where(object_mask)
        centroid = np.array([np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2])])
        
        object_metrics.append({
            'object_id': i,
            'volume': object_volume,
            'voxel_count': object_voxels,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1], 
            'centroid_z': centroid[2]
        })
    
    return {
        'total_volume': volume,
        'total_voxels': total_voxels,
        'surface_area': surface_area,
        'num_objects': num_objects,
        'object_metrics': object_metrics
    }

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        help="Upload a 3D image stack (TIFF) or single image"
    )
    
    if uploaded_file is not None:
        # Load image
        image_stack = load_image_stack(uploaded_file)
        if image_stack is not None:
            st.session_state.image_stack = image_stack
            st.success(f"✅ Loaded image stack: {image_stack.shape}")
            
            # Display image info
            st.write(f"**Image Info:**")
            st.write(f"- Shape: {image_stack.shape}")
            st.write(f"- Data type: {image_stack.dtype}")
            st.write(f"- Value range: {image_stack.min()} - {image_stack.max()}")

with col2:
    st.subheader("🔧 Processing Options")
    
    # Background correction
    if st.session_state.image_stack is not None:
        if st.button("🔄 Apply Background Correction"):
            with st.spinner("Applying background correction..."):
                corrected_stack = apply_background_correction(st.session_state.image_stack)
                st.session_state.corrected_stack = corrected_stack
                st.success("✅ Background correction applied!")
        
        # Brightness/Contrast controls
        st.subheader("🎛️ Image Adjustments")
        brightness = st.slider("Brightness", -100, 100, st.session_state.brightness, key="brightness_slider")
        contrast = st.slider("Contrast", 10, 300, st.session_state.contrast, key="contrast_slider")
        
        if st.button("Reset Adjustments"):
            st.session_state.brightness = 0
            st.session_state.contrast = 100
            st.rerun()
        
        # Show corrected/raw toggle
        show_corrected = st.checkbox("Show Background Corrected", st.session_state.show_corrected)

# Image display and analysis
if st.session_state.image_stack is not None:
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 2D Analysis", "🔬 Segmentation", "📐 3D Visualization", "📈 Metrics"])
    
    with tab1:
        st.subheader("2D Image Analysis")
        
        # Frame selection
        if len(st.session_state.image_stack.shape) == 3:
            frame_idx = st.slider("Frame", 0, st.session_state.image_stack.shape[0]-1, st.session_state.current_frame)
            st.session_state.current_frame = frame_idx
            
            # Get current frame
            current_frame = st.session_state.image_stack[frame_idx]
            
            # Apply corrections if available
            if st.session_state.corrected_stack is not None and show_corrected:
                current_frame = st.session_state.corrected_stack[frame_idx]
            
            # Apply brightness/contrast
            adjusted_frame = adjust_brightness_contrast(current_frame, brightness, contrast)
            
            # Display image
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original
            ax[0].imshow(current_frame, cmap='gray')
            ax[0].set_title("Original Frame")
            ax[0].axis('off')
            
            # Adjusted
            ax[1].imshow(adjusted_frame, cmap='gray')
            ax[1].set_title("Adjusted Frame")
            ax[1].axis('off')
            
            st.pyplot(fig)
            
            # Histogram
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            ax_hist.hist(current_frame.flatten(), bins=50, alpha=0.7, label='Original')
            if st.session_state.corrected_stack is not None and show_corrected:
                ax_hist.hist(adjusted_frame.flatten(), bins=50, alpha=0.7, label='Corrected')
            ax_hist.set_xlabel("Pixel Value")
            ax_hist.set_ylabel("Frequency")
            ax_hist.legend()
            ax_hist.set_title("Pixel Value Distribution")
            st.pyplot(fig_hist)
    
    with tab2:
        st.subheader("Advanced Segmentation")
        
        if st.session_state.image_stack is not None:
            # Segmentation method selection
            seg_method = st.selectbox(
                "Segmentation Method",
                ['threshold', 'otsu', 'adaptive', 'kmeans', 'watershed', 'ml_random_forest'],
                help="Choose the segmentation algorithm"
            )
            
            # Method-specific parameters
            if seg_method == 'threshold':
                threshold_val = st.slider("Threshold Value", 0, 255, 128)
            elif seg_method == 'kmeans':
                kmeans_clusters = st.slider("Number of Clusters", 2, 5, 3)
            elif seg_method == 'watershed':
                watershed_markers = st.slider("Number of Markers", 5, 20, 10)
            
            # Segmentation parameters
            col_seg1, col_seg2 = st.columns(2)
            
            with col_seg1:
                if st.button("🔍 Segment Current Frame"):
                    with st.spinner("Segmenting..."):
                        # Get current frame
                        current_frame = st.session_state.image_stack[st.session_state.current_frame]
                        if st.session_state.corrected_stack is not None and show_corrected:
                            current_frame = st.session_state.corrected_stack[st.session_state.current_frame]
                        
                        # Apply adjustments
                        adjusted_frame = adjust_brightness_contrast(current_frame, brightness, contrast)
                        
                        # Segment
                        if seg_method == 'threshold':
                            segmented = segment_slice_advanced(adjusted_frame, seg_method, threshold_val)
                        elif seg_method == 'kmeans':
                            segmented = segment_slice_advanced(adjusted_frame, seg_method, kmeans_clusters=kmeans_clusters)
                        elif seg_method == 'watershed':
                            segmented = segment_slice_advanced(adjusted_frame, seg_method, watershed_markers=watershed_markers)
                        else:
                            segmented = segment_slice_advanced(adjusted_frame, seg_method)
                        
                        # Display results
                        fig_seg, ax_seg = plt.subplots(1, 3, figsize=(15, 5))
                        
                        ax_seg[0].imshow(adjusted_frame, cmap='gray')
                        ax_seg[0].set_title("Input Image")
                        ax_seg[0].axis('off')
                        
                        ax_seg[1].imshow(segmented, cmap='gray')
                        ax_seg[1].set_title("Segmented")
                        ax_seg[1].axis('off')
                        
                        # Overlay
                        overlay = adjusted_frame.copy()
                        overlay[segmented > 0] = overlay[segmented > 0] // 2 + 128
                        ax_seg[2].imshow(overlay, cmap='gray')
                        ax_seg[2].set_title("Overlay")
                        ax_seg[2].axis('off')
                        
                        st.pyplot(fig_seg)
            
            with col_seg2:
                if st.button("🔍 Segment Entire Stack"):
                    with st.spinner("Processing entire stack..."):
                        # Process entire stack
                        segmented_stack = np.zeros_like(st.session_state.image_stack)
                        
                        for i in range(st.session_state.image_stack.shape[0]):
                            current_frame = st.session_state.image_stack[i]
                            if st.session_state.corrected_stack is not None and show_corrected:
                                current_frame = st.session_state.corrected_stack[i]
                            
                            adjusted_frame = adjust_brightness_contrast(current_frame, brightness, contrast)
                            
                            if seg_method == 'threshold':
                                segmented = segment_slice_advanced(adjusted_frame, seg_method, threshold_val)
                            elif seg_method == 'kmeans':
                                segmented = segment_slice_advanced(adjusted_frame, seg_method, kmeans_clusters=kmeans_clusters)
                            elif seg_method == 'watershed':
                                segmented = segment_slice_advanced(adjusted_frame, seg_method, watershed_markers=watershed_markers)
                            else:
                                segmented = segment_slice_advanced(adjusted_frame, seg_method)
                            
                            segmented_stack[i] = segmented
                        
                        st.session_state.segmented_stack = segmented_stack
                        st.success(f"✅ Segmented entire stack!")
    
    with tab3:
        st.subheader("3D Visualization")
        
        if st.session_state.segmented_stack is not None:
            # 3D visualization options
            surface_method = st.selectbox(
                "3D Visualization Method",
                ['marching_cubes', 'volume_rendering'],
                help="Choose 3D visualization approach"
            )
            
            if st.button("🎨 Generate 3D Visualization"):
                with st.spinner("Creating 3D visualization..."):
                    fig_3d = create_3d_surface_enhanced(st.session_state.segmented_stack, surface_method)
                    st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab4:
        st.subheader("Volume Analysis Metrics")
        
        if st.session_state.segmented_stack is not None:
            # Pixel size input
            pixel_size = st.number_input("Pixel Size (µm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            
            if st.button("📊 Calculate Metrics"):
                with st.spinner("Calculating metrics..."):
                    metrics = calculate_volume_metrics_enhanced(st.session_state.segmented_stack, pixel_size)
                    
                    # Display metrics
                    col_met1, col_met2 = st.columns(2)
                    
                    with col_met1:
                        st.metric("Total Volume", f"{metrics['total_volume']:.2f} µm³")
                        st.metric("Total Voxels", f"{metrics['total_voxels']:,}")
                        st.metric("Surface Area", f"{metrics['surface_area']:.2f} µm²")
                        st.metric("Number of Objects", metrics['num_objects'])
                    
                    with col_met2:
                        if metrics['object_metrics']:
                            st.write("**Object Analysis:**")
                            object_df = pd.DataFrame(metrics['object_metrics'])
                            st.dataframe(object_df, use_container_width=True)
                            
                            # Object volume distribution
                            fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
                            ax_dist.hist(object_df['volume'], bins=20, alpha=0.7)
                            ax_dist.set_xlabel("Volume (µm³)")
                            ax_dist.set_ylabel("Number of Objects")
                            ax_dist.set_title("Object Volume Distribution")
                            st.pyplot(fig_dist)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔬 <strong>3D Cell Analysis Tool - Enhanced Version</strong></p>
    <p>Advanced features: Background correction, multiple segmentation methods, brightness/contrast controls, 3D visualization, and comprehensive metrics</p>
    <p>For the full desktop experience with all features, download the original PyQt5 application</p>
    <p>📧 Contact: <a href="mailto:jalal.rahbani@hotmail.com">jalal.rahbani@hotmail.com</a> | 
    🌐 <a href="https://jalalrahbani.github.io/-jalal-website/" target="_blank">Personal Website</a></p>
</div>
""", unsafe_allow_html=True) 