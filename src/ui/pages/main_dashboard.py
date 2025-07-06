"""
Main Dashboard
==============

Main user interface for the Hands-Free Data Science application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MainDashboard:
    """Main dashboard for the application."""
    
    def __init__(self, data_manager, gesture_detector, voice_handler, ai_assistant):
        """
        Initialize the main dashboard.
        
        Args:
            data_manager: Data management instance
            gesture_detector: Gesture detection instance
            voice_handler: Voice recognition instance
            ai_assistant: AI assistant instance
        """
        self.data_manager = data_manager
        self.gesture_detector = gesture_detector
        self.voice_handler = voice_handler
        self.ai_assistant = ai_assistant
        
    def render(self):
        """Render the main dashboard."""
        
        # Page header
        st.title("🧠🖐️ Hands-Free Data Science")
        st.markdown("*Interact with datasets using your hands and voice*")
        
        # Sidebar for controls
        self._render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([3, 2])
        
        with col1:
            self._render_dataset_panel()
        
        with col2:
            self._render_interaction_panel()
        
        # Status bar
        self._render_status_bar()
    
    def _render_sidebar(self):
        """Render the sidebar with controls."""
        
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Dataset section
            st.subheader("📊 Dataset")
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=['csv', 'xlsx', 'json', 'parquet'],
                help="Upload a dataset to analyze"
            )
            
            if uploaded_file is not None:
                if st.button("Load Dataset"):
                    self._load_dataset(uploaded_file)
            
            # Sample datasets
            st.subheader("📝 Sample Datasets")
            sample_datasets = [
                "None",
                "Iris Dataset",
                "Titanic Dataset",
                "Boston Housing",
                "Wine Quality"
            ]
            
            selected_sample = st.selectbox(
                "Load Sample Dataset",
                sample_datasets
            )
            
            if selected_sample != "None" and st.button("Load Sample"):
                self._load_sample_dataset(selected_sample)
            
            # Voice controls
            st.subheader("🎤 Voice Controls")
            
            voice_enabled = st.checkbox("Enable Voice Recognition", value=False)
            
            if voice_enabled:
                if st.button("Start Listening"):
                    self._start_voice_recognition()
                
                if st.button("Stop Listening"):
                    self._stop_voice_recognition()
            
            # Camera controls
            st.subheader("📷 Camera Controls")
            
            camera_enabled = st.checkbox("Enable Hand Tracking", value=False)
            
            if camera_enabled:
                if st.button("Start Camera"):
                    self._start_camera()
                
                if st.button("Stop Camera"):
                    self._stop_camera()
            
            # Settings
            st.subheader("⚙️ Settings")
            
            if st.button("Reset Application"):
                self._reset_application()
    
    def _render_dataset_panel(self):
        """Render the dataset display panel."""
        
        st.header("📋 Dataset Viewer")
        
        # Check if dataset is loaded
        if self.data_manager.current_dataset is None:
            st.info("👋 Welcome! Please upload a dataset to get started.")
            
            # Show sample data options
            st.subheader("🎯 What you can do:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🖐️ Hand Gestures:**
                - Point to select data
                - Make selection boxes
                - Navigate through data
                """)
            
            with col2:
                st.markdown("""
                **🎙️ Voice Commands:**
                - "Show me the data"
                - "Visualize age distribution"
                - "Analyze correlation"
                """)
            
            return
        
        # Dataset info
        dataset_info = self.data_manager.get_dataset_info()
        
        # Display dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", dataset_info.get("shape", [0, 0])[0])
        
        with col2:
            st.metric("Columns", dataset_info.get("shape", [0, 0])[1])
        
        with col3:
            st.metric("Memory Usage", f"{dataset_info.get('memory_usage', 0) / 1024:.1f} KB")
        
        with col4:
            st.metric("Selected", len(self.data_manager.selected_rows) + len(self.data_manager.selected_columns))
        
        # Data display
        st.subheader("📊 Data Preview")
        
        # Show current selection if any
        if self.data_manager.selected_rows or self.data_manager.selected_columns:
            selected_data = self.data_manager.get_selected_data()
            if selected_data is not None:
                st.info(f"Showing selected data: {selected_data.shape[0]} rows, {selected_data.shape[1]} columns")
                st.dataframe(selected_data, use_container_width=True, height=300)
        else:
            # Show full dataset (limited)
            display_data = self.data_manager.current_dataset.head(100)
            st.dataframe(display_data, use_container_width=True, height=300)
        
        # Column statistics
        st.subheader("📈 Column Statistics")
        
        if dataset_info.get("columns"):
            selected_column = st.selectbox(
                "Select Column for Statistics",
                dataset_info["columns"]
            )
            
            if selected_column:
                column_stats = self.data_manager.get_column_statistics(selected_column)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json(column_stats)
                
                with col2:
                    # Quick visualization
                    if column_stats.get("dtype") in ["int64", "float64"]:
                        fig = px.histogram(
                            self.data_manager.current_dataset,
                            x=selected_column,
                            title=f"Distribution of {selected_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_interaction_panel(self):
        """Render the interaction panel."""
        
        st.header("🎯 Interaction Panel")
        
        # Voice command input
        st.subheader("🎤 Voice Command")
        
        # Manual text input for testing
        voice_command = st.text_input(
            "Enter command (or use voice)",
            placeholder="e.g., 'Show me the age distribution'"
        )
        
        if st.button("Process Command") and voice_command:
            self._process_voice_command(voice_command)
        
        # Recent commands
        st.subheader("📝 Recent Commands")
        
        if hasattr(self.ai_assistant, 'conversation_history') and self.ai_assistant.conversation_history:
            for i, conversation in enumerate(self.ai_assistant.conversation_history[-3:]):
                with st.expander(f"Command {len(self.ai_assistant.conversation_history) - i}"):
                    st.write(f"**User:** {conversation['user']}")
                    st.write(f"**Result:** {conversation['result'].get('explanation', 'No explanation')}")
        else:
            st.info("No recent commands. Try saying something!")
        
        # Camera feed placeholder
        st.subheader("📷 Camera Feed")
        
        camera_placeholder = st.empty()
        
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        if st.session_state.camera_active:
            camera_placeholder.info("📹 Camera is active - Hand tracking enabled")
        else:
            camera_placeholder.info("📷 Camera is inactive - Enable hand tracking in sidebar")
        
        # Visualization area
        st.subheader("📊 Visualizations")
        
        viz_placeholder = st.empty()
        
        # Show latest visualization if available
        if 'latest_visualization' in st.session_state:
            viz_placeholder.plotly_chart(st.session_state.latest_visualization, use_container_width=True)
        else:
            viz_placeholder.info("🎨 Visualizations will appear here")
    
    def _render_status_bar(self):
        """Render the status bar."""
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dataset_status = "✅ Loaded" if self.data_manager.current_dataset is not None else "❌ No Dataset"
            st.write(f"**Dataset:** {dataset_status}")
        
        with col2:
            voice_status = "🎤 Active" if st.session_state.get('voice_active', False) else "🔇 Inactive"
            st.write(f"**Voice:** {voice_status}")
        
        with col3:
            camera_status = "📹 Active" if st.session_state.get('camera_active', False) else "📷 Inactive"
            st.write(f"**Camera:** {camera_status}")
        
        with col4:
            ai_status = "🤖 Ready" if self.ai_assistant.is_initialized else "❌ Not Ready"
            st.write(f"**AI:** {ai_status}")
    
    def _load_dataset(self, uploaded_file):
        """Load uploaded dataset."""
        try:
            # Save uploaded file temporarily
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load dataset
            success = self.data_manager.load_dataset(tmp_file_path)
            
            if success:
                st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load dataset")
            
            # Clean up
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    def _load_sample_dataset(self, dataset_name):
        """Load sample dataset."""
        st.info(f"Loading sample dataset: {dataset_name}")
        # Placeholder for sample dataset loading
        
    def _process_voice_command(self, command):
        """Process voice command."""
        if self.data_manager.current_dataset is None:
            st.error("Please load a dataset first")
            return
        
        # Get dataset info
        dataset_info = self.data_manager.get_dataset_info()
        
        # Process command with AI
        result = self.ai_assistant.process_command(command, dataset_info)
        
        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(f"Command processed: {result.get('explanation', 'No explanation')}")
            
            # Execute visualization if requested
            if result.get("action") == "visualize":
                self._create_visualization(result.get("parameters", {}))
    
    def _create_visualization(self, parameters):
        """Create visualization based on parameters."""
        try:
            chart_type = parameters.get("chart_type", "histogram")
            
            if chart_type == "histogram":
                column = parameters.get("column")
                if column and column in self.data_manager.current_dataset.columns:
                    fig = px.histogram(
                        self.data_manager.current_dataset,
                        x=column,
                        title=parameters.get("title", f"Distribution of {column}")
                    )
                    st.session_state.latest_visualization = fig
                    st.rerun()
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    
    def _start_voice_recognition(self):
        """Start voice recognition."""
        st.session_state.voice_active = True
        st.success("Voice recognition started")
    
    def _stop_voice_recognition(self):
        """Stop voice recognition."""
        st.session_state.voice_active = False
        st.info("Voice recognition stopped")
    
    def _start_camera(self):
        """Start camera for hand tracking."""
        st.session_state.camera_active = True
        st.success("Camera started")
    
    def _stop_camera(self):
        """Stop camera."""
        st.session_state.camera_active = False
        st.info("Camera stopped")
    
    def _reset_application(self):
        """Reset the application state."""
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reset managers
        self.data_manager.current_dataset = None
        self.data_manager.selected_rows = []
        self.data_manager.selected_columns = []
        
        st.success("Application reset successfully")
        st.rerun() 