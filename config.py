import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Neo4j Configuration
    NEO4J_URI = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
    NEO4J_USER = os.environ.get('NEO4J_USER') or 'neo4j'
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD') or '12345678'
    NEO4J_DATABASE = os.environ.get('NEO4J_DATABASE') or 'neo4j'
    
    # Flask Configuration
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Data Files Configuration
    TOPOLOGY_FILE = os.environ.get('TOPOLOGY_FILE') or '河流拓扑结构.xlsx'
    WATER_QUALITY_FILE = os.environ.get('WATER_QUALITY_FILE') or '河道氨氮统计数据--环境容量.xlsx'
    TRAIN_DATA_FILE = os.environ.get('TRAIN_DATA_FILE') or 'train_tradedata.csv'
    TEST_DATA_FILE = os.environ.get('TEST_DATA_FILE') or 'test_tradedata.csv'
    
    # Machine Learning Configuration
    ML_MODEL_PATH = os.environ.get('ML_MODEL_PATH') or 'river_quality_model.pkl'
    ML_SCALER_PATH = os.environ.get('ML_SCALER_PATH') or 'river_quality_scaler.pkl'
    ML_RANDOM_STATE = int(os.environ.get('ML_RANDOM_STATE', 42))
    ML_TEST_SIZE = float(os.environ.get('ML_TEST_SIZE', 0.2))
    
    # Water Quality Configuration
    DEFAULT_ALERT_THRESHOLD = float(os.environ.get('DEFAULT_ALERT_THRESHOLD', 1.0))
    WATER_QUALITY_UNITS = os.environ.get('WATER_QUALITY_UNITS', 'mg/L')
    
    # System Configuration
    AUTO_REFRESH_INTERVAL = int(os.environ.get('AUTO_REFRESH_INTERVAL', 300))  # 5 minutes
    MAX_SEGMENTS = int(os.environ.get('MAX_SEGMENTS', 160))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'river_system.log')
    
    # Security Configuration
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Performance Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create upload folder if it doesn't exist
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
        
        # Create logs folder if it doesn't exist
        log_dir = os.path.dirname(Config.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    NEO4J_DATABASE = 'test'
    ML_MODEL_PATH = 'test_river_quality_model.pkl'
    ML_SCALER_PATH = 'test_river_quality_scaler.pkl'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}