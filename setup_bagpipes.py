#!/usr/bin/env python3
"""
Setup script for bagpipes to work in Streamlit Cloud environment.
Configures bagpipes to use a writable data directory.
"""
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_bagpipes_environment():
    """
    Configure bagpipes to use a writable directory for data files.
    This prevents PermissionError in read-only Streamlit Cloud environment.
    """
    try:
        # Create a writable directory for bagpipes data
        bagpipes_data_dir = os.path.join(tempfile.gettempdir(), 'bagpipes_data')
        os.makedirs(bagpipes_data_dir, exist_ok=True)
        
        # Set environment variable for bagpipes data directory
        os.environ['BAGPIPES_FILTERS'] = bagpipes_data_dir
        os.environ['BAGPIPES_DATA'] = bagpipes_data_dir
        
        logger.info(f"✅ Bagpipes data directory configured: {bagpipes_data_dir}")
        
        # Try to import bagpipes to trigger any initial setup
        logger.info("🔧 Testing bagpipes import...")
        import bagpipes as pipes
        logger.info("✅ Bagpipes imported successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup bagpipes: {e}")
        return False

if __name__ == "__main__":
    setup_bagpipes_environment()