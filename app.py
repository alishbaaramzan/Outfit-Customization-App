import streamlit as st
import torch
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

st.set_page_config(page_title="Outfit Customizer", layout="wide")

# Streamlit UI
st.title("Outfit Customizer")
st.write("Upload an image and customize parts of the outfit!")

# Initialize session state to store variables across reruns
if 'mask' not in st.session_state:
    st.session_state['mask'] = None
if 'img' not in st.session_state:
    st.session_state['img'] = None
if 'points' not in st.session_state:
    st.session_state['points'] = []
if 'generated_img' not in st.session_state:
    st.session_state['generated_img'] = None
if 'stage' not in st.session_state:
    st.session_state['stage'] = "upload"

# Function to convert PIL Image to bytes for display
def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# Function to handle click events
def handle_click(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        st.session_state['points'].append([x, y])
        st.session_state['last_point'] = [x, y]
        st.rerun()

# Install required packages when the app starts
@st.cache_resource
def install_dependencies():
    import subprocess
    with st.spinner("Installing dependencies (this may take a minute)..."):
        subprocess.check_call(["pip", "install", "diffusers", "transformers"])
    return True

# Load models (with caching to avoid reloading)
@st.cache_resource
def load_models():
    from transformers import SamModel, SamProcessor
    from diffusers import AutoPipelineForInpainting
    
    with st.spinner("Loading segmentation model..."):
        model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
        processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
    
    with st.spinner("Loading inpainting model..."):
        pipeline = AutoPipelineForInpainting.from_pretrained(
            "redstonehero/ReV_Animated_Inpainting", 
            torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload()
    
    return model, processor, pipeline

# Generate mask using SAM model
def generate_mask(img, point):
    model, processor, _ = load_models()
    
    input_points = [[point]]  # Format for processor
    
    # Process inputs and get outputs
    inputs = processor(img, input_points=input_points, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract and process masks
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    
    # Get the best mask
    if len(masks[0][0]) > 0:
        # Find the most suitable mask (typically the one with highest prediction score)
        mask_tensor = masks[0][0][0]  # Default to first mask
        to_pil = transforms.ToPILImage()
        binary_matrix = mask_tensor.to(dtype=torch.uint8)
        mask_pil = to_pil(binary_matrix * 255)
        return mask_pil
    
    return None

# Run inpainting with the selected mask
def run_inpainting(img, mask, prompt):
    _, _, pipeline = load_models()
    
    # Resize to appropriate dimensions
    img_resized = img.resize((512, 768))
    mask_resized = mask.resize((512, 768))
    
    # Run inpainting
    with st.spinner("Generating new design..."):
        result = pipeline(
            prompt=prompt,
            width=512,
            height=768,
            num_inference_steps=30,
            image=img_resized,
            mask_image=mask_resized,
            guidance_scale=3.0,
            strength=1.0
        ).images[0]
    
    return result

# Main app flow
if install_dependencies():
    # Step 1: Upload image
    if st.session_state['stage'] == "upload":
        uploaded_file = st.file_uploader("Choose an image of clothing to customize", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process the uploaded image
            st.session_state['img'] = Image.open(uploaded_file).convert("RGB")
            st.session_state['stage'] = "select_area"
            st.rerun()
    
    # Step 2: Select area to modify
    elif st.session_state['stage'] == "select_area":
        st.write("Click on the part of the outfit you want to modify (e.g., shirt, pants)")
        
        # Create a matplotlib figure for the interactive image
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(st.session_state['img'])
        ax.axis('on')
        
        # Display the image with onClick functionality
        clicked_coords = st.pyplot(fig, use_container_width=True)
        
        # Handle manual coordinate input
        col1, col2, col3 = st.columns(3)
        with col1:
            x_coord = st.number_input("X coordinate", 0, st.session_state['img'].width, 100)
        with col2:
            y_coord = st.number_input("Y coordinate", 0, st.session_state['img'].height, 100)
        with col3:
            if st.button("Set Point"):
                st.session_state['points'].append([x_coord, y_coord])
                st.session_state['last_point'] = [x_coord, y_coord]
        
        # Show selected points
        if 'last_point' in st.session_state:
            st.write(f"Selected point: {st.session_state['last_point']}")
            
            # Generate mask for the selected point
            mask = generate_mask(st.session_state['img'], st.session_state['last_point'])
            if mask:
                st.session_state['mask'] = mask
                
                # Display original and mask side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state['img'], caption="Original Image", use_column_width=True)
                with col2:
                    st.image(mask, caption="Selected Area (Mask)", use_column_width=True)
                
                # Option to proceed or select different area
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use this selection"):
                        st.session_state['stage'] = "customize"
                        st.rerun()
                with col2:
                    if st.button("Select different area"):
                        st.session_state['points'] = []
                        if 'last_point' in st.session_state:
                            del st.session_state['last_point']
                        if 'mask' in st.session_state:
                            st.session_state['mask'] = None
                        st.rerun()
    
    # Step 3: Customize with prompt
    elif st.session_state['stage'] == "customize":
        st.write("Enter a design prompt for the selected area")
        prompt = st.text_input("Design prompt (e.g., 'floral pattern', 'striped design', 'tactical style')", "floral pattern")
        
        # Show the selected area again
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state['img'], caption="Original Image", use_column_width=True)
        with col2:
            st.image(st.session_state['mask'], caption="Selected Area", use_column_width=True)
        
        # Generate button
        if st.button("Generate Design"):
            # Run inpainting
            result = run_inpainting(st.session_state['img'], st.session_state['mask'], prompt)
            st.session_state['generated_img'] = result
            st.session_state['stage'] = "result"
            st.rerun()
        
        # Back button
        if st.button("Back to area selection"):
            st.session_state['stage'] = "select_area"
            st.rerun()
    
    # Step 4: Show result
    elif st.session_state['stage'] == "result":
        st.write("Here's your customized outfit!")
        
        # Show before and after
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state['img'], caption="Original Image", use_column_width=True)
        with col2:
            st.image(st.session_state['generated_img'], caption="Customized Outfit", use_column_width=True)
        
        # Save option
        if st.download_button("Download Result", pil_to_bytes(st.session_state['generated_img']), file_name="customized_outfit.png"):
            st.success("Image saved successfully!")
        
        # Start over or modify more
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Over"):
                for key in st.session_state.keys():
                    if key != 'stage':
                        st.session_state[key] = None
                st.session_state['points'] = []
                st.session_state['stage'] = "upload"
                st.rerun()
        with col2:
            if st.button("Customize Another Part"):
                st.session_state['points'] = []
                if 'last_point' in st.session_state:
                    del st.session_state['last_point']
                st.session_state['mask'] = None
                st.session_state['stage'] = "select_area"
                st.rerun()

# Footer
st.markdown("---")
st.markdown("Outfit Customizer App - Powered by SlimSAM and Diffusion Models")