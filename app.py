import streamlit as st
import numpy as np
import pandas as pd
import cloudpickle
import os

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "pipe.pkl"
DATA_PATH = "laptop_data.csv"  # Make sure this file is in the same folder

# -------------------------------
# Load Model
# -------------------------------
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            pipe = cloudpickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model file (pipe.pkl): {e}")
        st.stop()
else:
    st.error("❌ Model file (pipe.pkl) not found. Please ensure it is uploaded or generated.")
    st.stop()

# -------------------------------
# Load Dataset for UI Options (needed for unique values)
# -------------------------------
# NOTE: The Streamlit app needs the raw 'laptop_data.csv' to get the unique
# values for the select boxes. The notebook should create a processed version,
# but for the UI we use the loaded file. The actual model pipeline
# handles the processing of the selected features.
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Apply the same pre-processing steps as in the notebook to get correct unique values for CPU/OS/GPU
        df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
        df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

        # Feature Engineering for CPU_BRAND (from notebook cells 39, 41, 42)
        def fetch(text):
            parts = text.split()
            if len(parts) >= 3:
                name = ' '.join(parts[0:3])
                if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
                    return name
                else:
                    if parts[0] == 'Intel':
                        return 'Other Intel Processor'
                    else:
                        return 'AMD processor'
            return 'Other Intel Processor' # Default safety
        
        df['CPU_NAME'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[0:3]))
        df['CPU_BRAND'] = df['CPU_NAME'].apply(fetch)

        # Feature Engineering for GPU_BRAND (from notebook cell 67)
        df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
        df = df[df['Gpu brand'] != 'ARM'] # Drop 'ARM' as in notebook cell 69

        # Feature Engineering for OS (from notebook cells 74, 75, 76)
        def cat_os(inp): 
            if inp == 'Windows 10' or inp == 'Windows 10 S' or inp == 'Windows 7':
                return 'Windows'
            elif inp == 'macOS' or inp == 'Mac OS X':
                return 'Mac'
            else:
                return 'Others/No OS/Linux'
        df['os'] = df['OpSys'].apply(cat_os)
        
    except Exception as e:
        st.error(f"❌ Error reading or processing dataset file ({DATA_PATH}): {e}")
        st.stop()

else:
    st.error("❌ Dataset file (laptop_data.csv) not found. Please upload it.")
    st.stop()

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("💻 Laptop Price Predictor")

# -------------------------------
# UI Elements
# -------------------------------
company = st.selectbox("Brand", sorted(df['Company'].unique()))
typename = st.selectbox("Type", sorted(df['TypeName'].unique()))
ram = st.selectbox("RAM (in GB)", sorted(df['Ram'].unique()))
weight = st.number_input("Weight of the Laptop (kg)", min_value=0.5, max_value=5.0, step=0.1)

touchscreen_opt = st.selectbox("Touchscreen", ['No', 'Yes'])
ips_opt = st.selectbox("IPS Display", ['No', 'Yes'])

screen_size = st.number_input("Screen Size (in inches)", min_value=10.0, max_value=20.0, step=0.1)
resolution = st.selectbox("Screen Resolution", [
    '1920x1080', '1366x768', '1600x900',
    '3840x2160', '3200x1800', '2880x1800',
    '2560x1600', '2560x1440', '2304x1440'
])

# NOTE: The notebook used 'CPU_BRAND' and 'Gpu brand' as the final features
# However, the UI options were loaded from the processed 'df' using 'CPU_BRAND' and 'Gpu brand'
cpu = st.selectbox("CPU", sorted(df['CPU_BRAND'].unique()))
hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1000, 2000]) # 1024/2048 in original code replaced with 1000/2000 for realistic UI/data alignment
ssd = st.selectbox("SSD (in GB)", [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 1000, 1024]) # Added more SSD options from notebook data
gpu = st.selectbox("GPU", sorted(df['Gpu brand'].unique()))
os_val = st.selectbox("OS", sorted(df['os'].unique()))

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    try:
        # Convert Yes/No to binary
        touchscreen = 1 if touchscreen_opt == 'Yes' else 0
        ips = 1 if ips_opt == 'Yes' else 0

        # Calculate PPI
        # Need to clean the resolution string to get X_res and Y_res
        try:
            # Check if resolution string contains 'x'
            if 'x' in resolution:
                X_res, Y_res = map(int, resolution.split('x'))
            else:
                # ScreenResolution column in laptop_data has other text. 
                # Need to extract resolution numbers. Using the notebook's logic as reference.
                # Assuming the resolution string contains 'X_res x Y_res'
                import re
                match = re.search(r'(\d+x\d+)', resolution)
                if match:
                    res_part = match.group(0)
                    X_res, Y_res = map(int, res_part.split('x'))
                else:
                    st.error("❌ Could not parse screen resolution.")
                    st.stop()
                    
            ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size
        except Exception:
            # Fallback if something goes wrong with resolution parsing
            st.error("❌ Error calculating PPI. Check Screen Resolution input.")
            st.stop()


        # NOTE: The notebook dropped 'ScreenResolution' and 'Memory', but the 
        # ColumnTransformer in the pipeline expects 12 features: 
        # [Company, TypeName, Ram, Weight, Touchscreen, IPS, PPI, CPU_BRAND, HDD, SSD, GPU_BRAND, OS]
        # The indices [0, 1, 7, 10, 11] for OneHotEncoder correspond to:
        # [Company(0), TypeName(1), CPU_BRAND(7), GPU_BRAND(10), OS(11)] 
        # The remaining columns (Ram, Weight, Touchscreen, IPS, PPI, HDD, SSD) are passed through.
        
        # To match the pipeline's expected input structure, create a DataFrame
        # with the features in the order they were in 'x' before dropping
        # 'ScreenResolution' and 'Memory' in the notebook.
        # Original 'x' columns after notebook cell 88:
        # 0: Company, 1: TypeName, 2: Ram, 3: Weight, 4: Touchscreen, 5: IPS, 
        # 6: PPI, 7: CPU_BRAND, 8: HDD, 9: SSD, 10: GPU_BRAND, 11: OS
        
        # Build DataFrame for prediction
        query_data = {
            'Company': [company],
            'TypeName': [typename],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [touchscreen],
            'IPS': [ips],
            'PPI': [ppi],
            'CPU_BRAND': [cpu], # Use the corrected name to match notebook's final feature 'x'
            'HDD': [hdd],
            'SSD': [ssd],
            'GPU_BRAND': [gpu], # Use the corrected name to match notebook's final feature 'x'
            'OS': [os_val] # Use the corrected name to match notebook's final feature 'x'
        }
        
        query_df = pd.DataFrame(query_data)
        
        # Prediction
        predicted_price_log = pipe.predict(query_df)[0]
        predicted_price = int(np.exp(predicted_price_log))
        st.success(f"💰 The predicted price of this laptop is: ₹ {predicted_price:,}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("⚠️ Check if the input feature names and order match the features used during model training.")
