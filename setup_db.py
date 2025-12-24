#!/usr/bin/env python3
"""
Setup script to initialize the PostgreSQL database for video search metadata.
Run this before starting the video search application.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_database():
    """Initialize the database and create tables"""
    try:
        from db_utils import get_db_manager
        
        print("Initializing database...")
        db_manager = get_db_manager()
        print("Database setup completed successfully!")
        
        # Test the connection
        stats = db_manager.get_video_stats()
        print(f"Database connection test successful. Current stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        print("\nPossible solutions:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your DATABASE_URL environment variable")
        print("3. Ensure the database exists and is accessible")
        print("4. Check that required packages are installed: pip install psycopg2-binary sqlalchemy")
        return False

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = ['DATABASE_URL']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these in your .env file or environment")
        print("Example DATABASE_URL: postgresql://user:password@localhost:5432/video_search_db")
        return False
    
    return True

if __name__ == "__main__":
    print("Video Search Database Setup")
    print("=" * 40)
    
    if not check_environment():
        sys.exit(1)
    
    if setup_database():
        print("\n✅ Database setup completed successfully!")
        print("You can now start the video search application.")
    else:
        print("\n❌ Database setup failed!")
        sys.exit(1)