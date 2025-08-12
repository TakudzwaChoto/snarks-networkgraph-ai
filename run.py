 #!/usr/bin/env python3
"""
River Water Quality Management System
Startup script for the Flask application
"""

import os
import sys
from app import app, river_system
from config import config

def main():
    """Main function to start the application"""
    
    # Get configuration from environment
    config_name = os.environ.get('FLASK_CONFIG', 'default')
    
    # Initialize configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Print startup information
    print("=" * 60)
    print("🌊 River Water Quality Management System")
    print("=" * 60)
    print(f"Configuration: {config_name}")
    print(f"Debug Mode: {app.config['DEBUG']}")
    print(f"Neo4j URI: {app.config['NEO4J_URI']}")
    print(f"Host: {app.config['HOST']}")
    print(f"Port: {app.config['PORT']}")
    print("=" * 60)
    
    # Check Neo4j connection
    if river_system.neo4j_connected:
        print("✅ Neo4j connection: SUCCESS")
    else:
        print("❌ Neo4j connection: FAILED")
        print("   Please ensure Neo4j is running and accessible")
        print("   Default credentials: neo4j/12345678")
    
    # Check ML model
    if river_system.ml_model is not None:
        print("✅ Machine Learning model: LOADED")
    else:
        print("⚠️  Machine Learning model: NOT LOADED")
        print("   Model will be trained on first use")
    
    print("=" * 60)
    print("🚀 Starting Flask application...")
    print(f"📱 Access the application at: http://{app.config['HOST']}:{app.config['PORT']}")
    print("=" * 60)
    
    try:
        # Start the Flask application
        app.run(
            host=app.config['HOST'],
            port=app.config['PORT'],
            debug=app.config['DEBUG']
        )
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()