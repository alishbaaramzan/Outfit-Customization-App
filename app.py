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

# Check for required packages when the app starts
@st.cache_resource
def check_dependencies():
    try:
        import importlib.util
        
        # Define required packages
        required_packages = ["diffusers", "transformers", "torch", "torchvision"]
        missing_packages = []
        
        # Check for each package
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        # Install missing packages if needed
        if missing_packages:
            import subprocess
            with st.spinner(f"Installing missing dependencies: {', '.join(missing_packages)}..."):
                try:
                    subprocess.check_call(["pip", "install"] + missing_packages)
                    st.success("Dependencies installed successfully!")
                except Exception as e:
                    st.error(f"Error installing dependencies: {e}")
                    st.error("Try refreshing the page or deploying again.")
                    return False
        
        return True
    except Exception as e:
        st.error(f"Error checking dependencies: {e}")
        return False

# Load models (with caching to avoid reloading)
@st.cache_resource
def load_models():
    from transformers import SamModel, SamProcessor
    from diffusers import AutoPipelineForInpainting
    
    with st.spinner("Loading segmentation model..."):
        try:
            # Use a smaller, more lightweight SAM model
            model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50", 
                                             low_cpu_mem_usage=True, 
                                             variant="cpu")
            processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
        except Exception as e:
            st.error(f"Error loading segmentation model: {e}")
            st.error("Try refreshing the page or using a different model.")
            raise e
    
    with st.spinner("Loading inpainting model..."):
        try:
            # Use a smaller inpainting model with CPU optimization
            pipeline = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting",  # Using standard SD inpainting model instead of larger one
                torch_dtype=torch.float32,  # Use float32 for better compatibility
                use_safetensors=True,
                variant="fp16",
                low_cpu_mem_usage=True
            )
            # Don't use model_cpu_offload on Streamlit Cloud as it can cause issues
        except Exception as e:
            st.error(f"Error loading inpainting model: {e}")
            st.error("Try refreshing the page or using a different model.")
            raise e
    
    return model, processor, pipeline

# Generate mask using SAM model
def generate_mask(img, point):
    try:
        model, processor, _ = load_models()
        
        # Display coordinate information
        st.info(f"Selected point: {point}. For reference, image dimensions are {img.width}x{img.height}")
        st.info("If the mask doesn't highlight the desired area, try selecting a different point on the garment.")
        
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
            # Try multiple masks if available
            mask_idx = 0
            if len(masks[0][0]) > 1:
                # Use the second mask if available as it's often better for clothing items
                mask_idx = 1
            
            mask_tensor = masks[0][0][mask_idx]
            to_pil = transforms.ToPILImage()
            binary_matrix = mask_tensor.to(dtype=torch.uint8)
            mask_pil = to_pil(binary_matrix * 255)
            
            st.success(f"Found {len(masks[0][0])} potential masks. Using mask #{mask_idx+1}.")
            return mask_pil
        
        st.warning("No mask found. Try selecting a different point on the garment.")
        return None
    except Exception as e:
        st.error(f"Error generating mask: {e}")
        return None

# Run inpainting with the selected mask
def run_inpainting(img, mask, prompt):
    try:
        _, _, pipeline = load_models()
        
        # Resize to more memory-efficient dimensions
        target_width = 384  # Smaller size for efficiency
        target_height = 512
        img_resized = img.resize((target_width, target_height))
        mask_resized = mask.resize((target_width, target_height))
        
        # Run inpainting with optimized parameters
        with st.spinner("Generating new design (this might take a while)..."):
            result = pipeline(
                prompt=prompt,
                width=target_width,
                height=target_height,
                num_inference_steps=20,  # Reduced steps for speed
                image=img_resized,
                mask_image=mask_resized,
                guidance_scale=7.5,      # Standard guidance scale
                num_images_per_prompt=1,
                negative_prompt="low quality, blurry, distorted, deformed",
            ).images[0]
        
        return result
    except Exception as e:
        st.error(f"Error during inpainting: {e}")
        st.error("The model may have run out of memory. Try with a smaller image or a simpler prompt.")
        return None

# Set page config
st.set_page_config(
    page_title="Outfit Customizer", 
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add sidebar with instructions
with st.sidebar:
    st.title("📋 Instructions")
    st.markdown("""
    ### How to use this app:
    
    1. **Upload an image** of clothing you want to customize
    2. **Select an area** on the garment you want to modify
    3. **Enter a design prompt** describing what you want
    4. **Generate** the new design
    
    ### Tips for good results:
    
    - Use clear images with solid backgrounds
    - Select points in the center of the garment area
    - Use descriptive prompts like "floral pattern" or "plaid design"
    - If results aren't good, try selecting a different point
    
    ### About:
    
    This app uses AI segmentation and diffusion models to customize clothing items in images.
    """)
    
    st.markdown("---")
    st.caption("⚠️ This app runs in the browser and may take some time to load and process images.")

# Main app flow
if check_dependencies():
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
        st.write("Select the part of the outfit you want to modify (e.g., shirt, pants)")
        
        # Create a container for the image display
        image_container = st.container()
        
        # Display the image
        with image_container:
            st.image(st.session_state['img'], caption="Click a specific part (like shirt or pants)", use_column_width=True)
            
        # Add a guide for good selection points
        st.info("👕 **Selection Guide:**")
        st.markdown("""
        - For shirts/tops: Select a point in the middle of the chest area
        - For pants/bottoms: Select a point in the middle of the thigh area
        - For sleeves: Select mid-sleeve
        - Try to click near the center of the item you want to change
        """)
        
        # Handle coordinate input with examples
        st.write("### Enter Coordinates")
        st.write("Enter the X,Y position of the part you want to change:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Default to center of image as starting point
            default_x = st.session_state['img'].width // 2
            x_coord = st.number_input("X coordinate", 0, st.session_state['img'].width, default_x)
        with col2:
            default_y = st.session_state['img'].height // 2
            y_coord = st.number_input("Y coordinate", 0, st.session_state['img'].height, default_y)
        with col3:
            if st.button("Set Point", key="set_point_btn"):
                st.session_state['points'].append([x_coord, y_coord])
                st.session_state['last_point'] = [x_coord, y_coord]
                st.rerun()
        
        # Quick selection buttons for common garment areas
        st.write("### Quick Selection")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Select Top/Shirt", key="select_top"):
                # Approximate coordinates for upper body
                x = st.session_state['img'].width // 2
                y = st.session_state['img'].height // 3
                st.session_state['last_point'] = [x, y]
                st.rerun()
        with col2:
            if st.button("Select Bottom/Pants", key="select_bottom"):
                # Approximate coordinates for lower body
                x = st.session_state['img'].width // 2
                y = int(st.session_state['img'].height * 0.7)
                st.session_state['last_point'] = [x, y]
                st.rerun()
        with col3:
            if st.button("Select Sleeves", key="select_sleeves"):
                # Approximate coordinates for sleeve
                x = int(st.session_state['img'].width * 0.25)
                y = int(st.session_state['img'].height * 0.4)
                st.session_state['last_point'] = [x, y]
                st.rerun()
        
        # Show selected points
        if 'last_point' in st.session_state:
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
                    if st.button("Use this selection", key="use_selection"):
                        st.session_state['stage'] = "customize"
                        st.rerun()
                with col2:
                    if st.button("Select different area", key="diff_selection"):
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