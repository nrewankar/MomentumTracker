import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Import our modules
from momentum_calculator import calculate_momentum_scores, get_top_bottom_stocks
from data_loader import load_ticker_data, format_momentum_data, get_industry_breakdown
from visualizations import (
    plot_momentum_distribution, 
    plot_industry_momentum, 
    plot_top_bottom_momentum,
    plot_industry_breakdown,
    plot_momentum_heatmap
)

# Set page config without sidebar
st.set_page_config(
    page_title="S&P 500 Momentum Factor Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a cleaner, more modern look
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    h1, h2, h3 {
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        background-color: white;
    }
    div[data-testid="stCheckbox"] > label > div[data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
    }
    button[data-testid="baseButton-primary"] {
        background-color: #1E88E5;
        font-weight: 500;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    div[data-testid="stForm"] {
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        background-color: white;
        padding: 1rem;
    }
    div.element-container {
        margin-bottom: 1rem;
    }
    .plot-container {
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        background-color: white;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing calculation results
if 'momentum_results' not in st.session_state:
    st.session_state.momentum_results = None
if 'formatted_data' not in st.session_state:
    st.session_state.formatted_data = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'using_custom_data' not in st.session_state:
    st.session_state.using_custom_data = False
if 'calculation_status' not in st.session_state:
    st.session_state.calculation_status = None

# Store last updated time per data source
if 'last_updated_default' not in st.session_state:
    st.session_state.last_updated_default = None
if 'last_updated_custom' not in st.session_state:
    st.session_state.last_updated_custom = None

# For backward compatibility with previous sessions
if 'last_updated' in st.session_state and st.session_state.last_updated is not None:
    # Copy the old single timestamp to the appropriate source
    if st.session_state.data_source == "default":
        st.session_state.last_updated_default = st.session_state.last_updated
    else:
        st.session_state.last_updated_custom = st.session_state.last_updated
    # Clear the old value to avoid confusion
    st.session_state.last_updated = None

def calculate_momentum(use_cache=True, custom_file=None):
    """Calculate momentum scores and update session state"""
    # Verify data source based on our new session state approach
    if st.session_state.data_source == "default":
        custom_file = None
        data_source = "S&P 500 stocks"
        st.session_state.using_custom_data = False
    else:
        # Ensure using_custom_data flag is consistent with data_source
        st.session_state.using_custom_data = True
        
        # Make sure we have a file when using custom data
        if custom_file is None:
            st.warning("No custom file selected. Please upload a CSV file with ticker symbols.")
            st.info("To proceed, upload a file using the file uploader below.")
            return
        data_source = "custom stocks"
    
    # Enhanced logging for easier debugging
    print(f"MOMENTUM: Calculating for data source: {data_source}")
    print(f"MOMENTUM: Session state data_source: {st.session_state.data_source}")
    print(f"MOMENTUM: Using custom data flag: {st.session_state.using_custom_data}")
    print(f"MOMENTUM: Custom file provided: {custom_file is not None}")
    
    # Update calculation status in session state
    st.session_state.calculation_status = "calculating"
    
    # Create status containers for displaying different stages of processing
    main_status_container = st.empty()
    progress_container = st.empty()
    detail_container = st.empty()
    
    # Show the initial calculating message
    main_status_container.info(f"📊 Calculating momentum scores for {data_source}...")
    progress_container.markdown("""
    <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
        <p style="margin: 0; font-size: 0.9rem;">
            <b>Step 1/4:</b> Preparing to download stock price data...
        </p>
    </div>
    """, unsafe_allow_html=True)
    detail_container.caption("This may take several minutes. The app needs to download historical price data and process it.")
    
    with st.spinner(f"Processing stock data... This may take a few minutes."):
        # Get date range
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')  # Use 2 years of data
        
        # Add a debug expander to show calculation parameters
        with st.expander("Calculation Parameters", expanded=False):
            st.code(f"""
Data Source: {data_source}
Using Custom Data: {st.session_state.using_custom_data}
Start Date: {start_date}
End Date: {end_date}
Using Cache: {use_cache}
Custom File: {'Provided' if custom_file is not None else 'None'}
            """)
        
        # Make sure we reset the file pointer if we're using a custom file
        if custom_file is not None:
            try:
                custom_file.seek(0)
            except:
                main_status_container.error("Error accessing the uploaded file. Please try uploading it again.")
                st.session_state.calculation_status = "error"
                return
        
        # Before calculating, update the progress indicator to step 2
        progress_container.markdown("""
        <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 0.9rem;">
                <b>Step 2/4:</b> Downloading historical price data from Yahoo Finance...
            </p>
        </div>
        """, unsafe_allow_html=True)
        detail_container.caption("Downloading data for 500+ stocks can take several minutes. The app is downloading in small batches to avoid API rate limits.")
        
        # Set a time limit for the calculation
        import threading
        import time
        
        # Create a flag to track if the calculation is taking too long
        calculation_timeout = False
        
        # Function to update the UI during long calculations
        def update_waiting_message():
            wait_time = 0
            max_wait_time = 180  # 3 minutes max wait time
            update_interval = 20  # Update message every 20 seconds
            
            nonlocal calculation_timeout
            
            while wait_time < max_wait_time and not calculation_timeout:
                time.sleep(update_interval)
                wait_time += update_interval
                
                if wait_time >= 60:  # After 1 minute
                    detail_container.caption(f"Still working... This can take several minutes. Downloaded data is being processed (waited {wait_time} seconds)")
                
                if wait_time >= 120:  # After 2 minutes
                    progress_container.markdown("""
                    <div style="padding: 10px; border-radius: 5px; background-color: #fff3cd; margin-bottom: 10px;">
                        <p style="margin: 0; font-size: 0.9rem;">
                            <b>Taking longer than expected:</b> Processing large amount of data. Please continue to wait...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # If we hit the max wait time, show a timeout message
            if wait_time >= max_wait_time and not calculation_timeout:
                calculation_timeout = True
                progress_container.markdown("""
                <div style="padding: 10px; border-radius: 5px; background-color: #f8d7da; margin-bottom: 10px;">
                    <p style="margin: 0; font-size: 0.9rem;">
                        <b>Taking too long:</b> The calculation may be stuck due to API rate limiting.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                detail_container.caption("Try using cached data or try again later. Yahoo Finance API may be throttling requests.")
        
        # Start the timer thread
        timer_thread = threading.Thread(target=update_waiting_message)
        timer_thread.daemon = True  # This ensures the thread won't prevent app from exiting
        timer_thread.start()
        
        try:
            # Calculate momentum with caching option (with timeout handling)
            results = calculate_momentum_scores(
                start_date=start_date, 
                end_date=end_date, 
                use_cache=use_cache,
                custom_file=custom_file
            )
            # Mark that we've finished calculation
            calculation_timeout = True
        except Exception as e:
            # Mark that we've finished calculation
            calculation_timeout = True
            # Show error message
            main_status_container.error(f"Error during calculation: {str(e)}")
            detail_container.info("There was an unexpected error. Try again with 'Use cached data' option enabled.")
            st.session_state.calculation_status = "error"
            return
        
        # Check if we have partial results before marking as error
        if "error" not in results:
            # Update progress to step 3 - data processing
            progress_container.markdown("""
            <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                <p style="margin: 0; font-size: 0.9rem;">
                    <b>Step 3/4:</b> Calculating momentum scores for each stock...
                </p>
            </div>
            """, unsafe_allow_html=True)
            detail_container.caption("Analyzing price data to compute momentum factors, ranking stocks, and preparing visualizations...")
        
            # After we get results and before we format, update to step 4
            progress_container.markdown("""
            <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                <p style="margin: 0; font-size: 0.9rem;">
                    <b>Step 4/4:</b> Finalizing analysis and preparing dashboard...
                </p>
            </div>
            """, unsafe_allow_html=True)
            detail_container.caption("Almost done! Formatting results and preparing interactive visualizations...")
        
        if "error" not in results:
            st.session_state.momentum_results = results
            st.session_state.formatted_data = format_momentum_data(results)
            
            # Update the appropriate last updated timestamp based on data source
            current_time = datetime.now()
            if st.session_state.data_source == "default":
                st.session_state.last_updated_default = current_time
            else:
                st.session_state.last_updated_custom = current_time
            
            # Clear the previous status messages and show success
            main_status_container.empty()
            progress_container.empty()
            detail_container.empty()
            main_status_container.success("Momentum calculation completed successfully!")
            st.session_state.calculation_status = "complete"
        else:
            # Clear the previous status messages and show error
            main_status_container.empty()
            progress_container.empty()
            detail_container.empty()
            main_status_container.error(f"Error calculating momentum: {results['error']}")
            detail_container.info("Consider using a smaller data set or enabling data caching to avoid API rate limiting.")
            st.session_state.calculation_status = "error"

# Header section with modern styling
st.markdown("""
<div style="background-color:#1E88E5; padding:10px; border-radius:10px; margin-bottom:20px;">
    <h1 style="color:white; text-align:center; margin:0;">Stock Momentum Factor Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# Control panel in main layout
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <h2 style="color: #424242; margin: 0;">Control Panel</h2>
</div>
""", unsafe_allow_html=True)

# Data Source Section
st.markdown("""
<div style="margin-bottom: 15px;">
    <h3 style="color: #424242; font-size: 1.2rem; margin-bottom: 10px;">Data Source</h3>
</div>
""", unsafe_allow_html=True)

# Complete redesign of the data source selection mechanism
# More direct approach with simplified state management

# Function to handle selecting the default S&P 500 list
def use_default_sp500():
    """Switch to using the default S&P 500 list"""
    st.session_state.data_source = "default"
    st.session_state.using_custom_data = False
    st.session_state.uploaded_file = None
    
    # Clear previous results when switching data sources
    if 'momentum_results' in st.session_state:
        st.session_state.momentum_results = None
        st.session_state.formatted_data = None
    
    print("DATA SOURCE: Switched to S&P 500 default list")

# Function to handle selecting custom ticker list
def use_custom_list():
    """Switch to using a custom uploaded list"""
    st.session_state.data_source = "custom"
    st.session_state.using_custom_data = True
    print("DATA SOURCE: Switched to custom ticker list")

# Initialize the data source state if not present
if 'data_source' not in st.session_state:
    st.session_state.data_source = "default"
    st.session_state.using_custom_data = False

# Create radio buttons without callback - using direct click handlers instead
col1, col2 = st.columns(2)

with col1:
    sp500_selected = st.button(
        "📊 Use S&P 500 List", 
        type="primary" if st.session_state.data_source == "default" else "secondary",
        use_container_width=True,
        on_click=use_default_sp500
    )
    
with col2:
    custom_selected = st.button(
        "📁 Use Custom Ticker List", 
        type="primary" if st.session_state.data_source == "custom" else "secondary",
        use_container_width=True,
        on_click=use_custom_list
    )

# Display which data source is currently selected (banner style notification)
if st.session_state.data_source == "default":
    st.success("**Using S&P 500 default list** - The analysis will use the built-in S&P 500 constituent list")
else:
    st.info("**Using custom ticker list** - Please upload your CSV file below")

# Show appropriate content based on selection
if st.session_state.data_source == "default":
    st.write("Using the default S&P 500 constituent list included with the application.")
    
    # Add a debug display of the current state
    with st.expander("Debug State Information", expanded=False):
        st.code(f"""
using_custom_data: {st.session_state.using_custom_data}
uploaded_file: {'Present' if st.session_state.uploaded_file is not None else 'None'}
data_source: {st.session_state.data_source}
        """)

else:  # Custom list selected
    st.write("Upload your own CSV file with ticker symbols.")
    st.markdown("""
    <div style="padding: 10px; border-radius: 5px; background-color: #e3f2fd; margin-bottom: 10px;">
        <p style="margin: 0; font-size: 0.9rem;">
            Your CSV file must include a <code>Symbol</code> column with ticker symbols. 
            Optional columns include <code>Company</code>, <code>Industry</code>, and <code>Year_Added</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload ticker list (CSV)", 
        type=["csv"], 
        help="CSV file with ticker symbols",
        key="ticker_file_uploader"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file upload
        is_new_file = True
        if st.session_state.uploaded_file is not None:
            # Try to determine if it's a different file
            current_file = st.session_state.uploaded_file
            try:
                current_file.seek(0)
                new_file_content = uploaded_file.read()
                uploaded_file.seek(0)  # Reset pointer
                
                current_file_content = current_file.read()
                current_file.seek(0)  # Reset pointer
                
                is_new_file = new_file_content != current_file_content
            except:
                # If any error occurs in comparison, treat as a new file
                is_new_file = True
        
        # Store the file and update state
        st.session_state.uploaded_file = uploaded_file
        st.session_state.using_custom_data = True
        
        # If it's a new file, clear previous results
        if is_new_file and 'momentum_results' in st.session_state:
            st.session_state.momentum_results = None
            st.session_state.formatted_data = None
            
        # Preview the uploaded file
        try:
            df_preview = pd.read_csv(uploaded_file)
            
            if is_new_file:
                st.success(f"Successfully loaded new file with {len(df_preview)} ticker symbols.")
                st.info("Click 'Calculate Momentum Scores' to analyze your custom ticker list.")
            else:
                st.success(f"File loaded with {len(df_preview)} ticker symbols.")
                
            st.write("Preview of your data:")
            st.dataframe(df_preview.head(5), use_container_width=True)
            
            # Reset the file pointer for later use
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.uploaded_file = None
            st.session_state.using_custom_data = False
            
        # Add a debug display of the current state
        with st.expander("Debug State Information", expanded=False):
            try:
                df_size = len(df_preview)
            except:
                df_size = 'Unknown'
                
            st.code(f"""
using_custom_data: {st.session_state.using_custom_data}
uploaded_file: {'Present' if st.session_state.uploaded_file is not None else 'None'}
custom_file_size: {df_size} rows
            """)
    else:
        st.warning("Please upload a CSV file with ticker symbols to use custom data.")
        
        # If no file is uploaded but custom data is selected, show a warning
        if st.session_state.using_custom_data and st.session_state.uploaded_file is None:
            st.error("No file is currently uploaded. Please upload a file or switch to S&P 500 data.")
            
            # Add a note about using the buttons above
            st.info("To switch to the S&P 500 data, click the 'Use S&P 500 List' button at the top of this page.")

# Create a horizontal layout for controls
st.markdown("""
<div style="margin: 20px 0 15px 0;">
    <h3 style="color: #424242; font-size: 1.2rem; margin-bottom: 10px;">Calculation Options</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    # Add caching option with improved styling
    use_cache = st.checkbox("Use cached data if available", value=True, 
                           help="Use previously downloaded data to avoid rate limiting")

# No momentum factor explanation as requested

with col2:
    # Create two sub-columns within col2
    col2_1, col2_2 = st.columns([3, 2])
    
    with col2_1:
        # Add calculation button with custom styling
        calculate_button = st.button("Calculate Momentum Scores", 
                                  use_container_width=True, 
                                  type="primary")
        if calculate_button:
            # Determine which file to use
            custom_file = st.session_state.uploaded_file if st.session_state.using_custom_data else None
            calculate_momentum(use_cache=use_cache, custom_file=custom_file)
    
    with col2_2:
        # Show last updated time based on the current data source
        current_source = "default" if st.session_state.data_source == "default" else "custom"
        last_updated_time = st.session_state.last_updated_default if current_source == "default" else st.session_state.last_updated_custom
        
        if last_updated_time:
            data_source_text = "S&P 500" if current_source == "default" else "Custom Data"
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; text-align: center;">
                <p style="color: #1976d2; margin: 0; font-size: 0.9rem;">Last updated ({data_source_text})</p>
                <p style="margin: 0; font-weight: bold;">{last_updated_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No data loaded yet for this source.")

# Add a divider
st.markdown("<hr style='margin: 20px 0; border: none; height: 1px; background-color: #e0e0e0;'>", unsafe_allow_html=True)

# Loading sample data if not calculated yet
if st.session_state.momentum_results is None:
    st.markdown("""
    <div style="padding: 25px; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h2 style="margin-top:0; color: #1E88E5;">Welcome to the Stock Momentum Dashboard</h2>
        <p style="font-size: 1.1rem; margin-bottom: 20px;">
            This dashboard analyzes momentum factors for stocks and provides interactive visualizations to help you identify market trends.
        </p>
        <p style="font-size: 1rem; margin-bottom: 20px;">
            You can use the default S&P 500 list or upload your own custom ticker list in CSV format.
        </p>
        <p style="font-weight: bold; color: #424242;">
            To get started, select your data source and click the "Calculate Momentum Scores" button above.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add note about Yahoo Finance API limitations
    st.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color: #fffde7; border-left: 5px solid #ffc107; margin-bottom: 20px;">
        <h3 style="margin-top:0; color: #ff9800;">Note about data retrieval</h3>
        <p>This application uses Yahoo Finance API which has rate limiting. For best results:</p>
        <ul>
            <li>Make sure the "Use cached data if available" option is checked</li>
            <li>If you encounter errors, wait a few minutes before trying again</li>
            <li>The app will automatically handle missing symbols and use available data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
        
    st.stop()  # Stop execution until calculation is performed

# If we have data, display the dashboard
if st.session_state.formatted_data and "error" not in st.session_state.formatted_data:
    formatted_data = st.session_state.formatted_data
    
    # Metadata about the calculation
    st.markdown("""
    <h2 style="color: #424242; border-bottom: 2px solid #1E88E5; padding-bottom: 8px; margin-bottom: 20px;">Overview</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Style the metrics with custom HTML/CSS
    with col1:
        st.markdown(f"""
        <div class="metric-container" style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f7ff; border-left: 5px solid #1E88E5;">
            <p style="color: #616161; font-size: 0.9rem; margin-bottom: 5px;">TOTAL STOCKS ANALYZED</p>
            <h2 style="color: #1E88E5; font-size: 2rem; margin: 0;">{len(formatted_data["display_df"])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Format the date for display
        last_date = pd.to_datetime(formatted_data["last_date"]).strftime("%Y-%m-%d")
        st.markdown(f"""
        <div class="metric-container" style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f7ff; border-left: 5px solid #1E88E5;">
