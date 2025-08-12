 # River Water Quality Management System

A comprehensive web application for managing river water quality using Neo4j graph database and machine learning predictions. This system provides real-time monitoring, predictive analytics, and interactive visualizations for river network topology and water quality parameters.

## 🌊 Features

### Core Functionality
- **Neo4j Graph Database Integration**: Advanced graph database for complex river topology relationships
- **Machine Learning Predictions**: AI-powered water quality trading predictions using Random Forest
- **Real-time Monitoring**: Live monitoring of water quality parameters and alerts
- **Interactive Network Visualization**: Dynamic graph visualization of river segments and connections
- **Water Quality Alerts**: Automated alert system for threshold violations
- **Trading Analysis**: Water quality trading simulation and optimization

### Technical Features
- **Modern Web UI**: Responsive design with Bootstrap 5 and custom CSS
- **Interactive Charts**: Plotly.js integration for advanced data visualization
- **RESTful API**: Clean API endpoints for data access and predictions
- **Real-time Updates**: Automatic data refresh and live monitoring
- **Mobile Responsive**: Optimized for desktop and mobile devices

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flask Web     │    │   Neo4j Graph   │    │   Machine       │
│   Application   │◄──►│   Database      │    │   Learning      │
│                 │    │                 │    │   Model         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Interactive   │    │   River         │    │   Prediction    │
│   Dashboard     │    │   Topology      │    │   Engine        │
│                 │    │   Data          │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

Before running this application, ensure you have:

1. **Python 3.8+** installed
2. **Neo4j Database** running locally or remotely
3. **Required Excel files** in the project directory:
   - `河流拓扑结构.xlsx` (River Topology Structure)
   - `河道氨氮统计数据--环境容量.xlsx` (Ammonia Nitrogen Statistics)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd river-management-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Neo4j
1. Install Neo4j Desktop or Neo4j Community Edition
2. Create a new database
3. Set username: `neo4j` and password: `12345678`
4. Start the Neo4j service

### 4. Prepare Data Files
Ensure the following Excel files are in the project root:
- `河流拓扑结构.xlsx` - Contains river segment topology data
- `河道氨氮统计数据--环境容量.xlsx` - Contains water quality statistics

### 5. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📊 Data Structure

### River Topology Data (`河流拓扑结构.xlsx`)
- **Subbasin**: River segment identifier
- **FROM_NODE**: Source node ID
- **TO_NODE**: Target node ID
- **FLOW_OUTcms**: Flow rate in cubic meters per second
- **AreaC**: Catchment area
- **Len2**: River length
- **Slo2**: River slope
- **Wid2**: River width
- **Dep2**: River depth

### Water Quality Data (`河道氨氮统计数据--环境容量.xlsx`)
- **RCH**: River segment identifier (matches Subbasin)
- **Cs**: Ammonia nitrogen concentration
- **K**: Record identifier

## 🔧 API Endpoints

### Graph Data
- `GET /api/graph-data` - Retrieve river network graph data
- `GET /api/water-quality-stats` - Get water quality statistics
- `GET /api/water-quality-alerts?threshold=1.0` - Get water quality alerts

### Predictions
- `POST /api/predict-trade` - Predict water quality trading amount
  ```json
  {
    "buyer": 1,
    "seller": 2
  }
  ```

### Database Management
- `GET /initialize-database` - Initialize Neo4j database with river data

## 🎯 Usage Guide

### 1. Initial Setup
1. Start the application
2. Click "Initialize Database" to load river data into Neo4j
3. Wait for the database initialization to complete

### 2. Dashboard Overview
- **Statistics Cards**: View key metrics like total segments, connections, and alerts
- **Network Graph**: Interactive visualization of river topology
- **Water Quality Trends**: Historical data and trends
- **Real-time Alerts**: Current water quality violations

### 3. Making Predictions
1. Navigate to the Predictions section
2. Enter buyer and seller segment numbers (1-160)
3. Click "Predict Trade Amount" to get ML predictions
4. View predicted water quality trading amount

### 4. Monitoring Alerts
1. Set alert threshold for ammonia nitrogen concentration
2. Click "Check Alerts" to view current violations
3. Monitor real-time updates in the dashboard

## 🧠 Machine Learning Model

### Model Details
- **Algorithm**: Random Forest Regressor
- **Features**: 
  - Buyer and seller segment IDs
  - Distance between segments
  - Derived segment characteristics
- **Target**: Water quality trading amount (WEC units)
- **Training Data**: Historical trading data from `train_tradedata.csv`

### Model Performance
- **Accuracy**: ~87.5% (based on historical data)
- **Features**: 5 engineered features
- **Training**: Automatic retraining when new data is available

## 🎨 UI Components

### Main Dashboard
- **Hero Section**: System overview and quick actions
- **Statistics Cards**: Key performance indicators
- **Network Visualization**: Interactive river topology graph
- **Prediction Interface**: ML-powered trading predictions
- **Alert System**: Real-time water quality monitoring

### Analytics Dashboard
- **Trend Charts**: Water quality over time
- **Distribution Analysis**: Segment quality distribution
- **Trading Analysis**: Volume and patterns
- **Performance Metrics**: System performance indicators
- **Top Segments**: Best performing river segments

## 🔒 Security Considerations

- **Database Authentication**: Secure Neo4j connection with credentials
- **Input Validation**: All user inputs are validated
- **Error Handling**: Comprehensive error handling and logging
- **Data Sanitization**: All data is properly sanitized before processing

## 🚨 Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Ensure Neo4j is running on `localhost:7687`
   - Verify username: `neo4j` and password: `12345678`
   - Check firewall settings

2. **Missing Data Files**
   - Ensure Excel files are in the project root directory
   - Verify file names match exactly
   - Check file permissions

3. **ML Model Training Issues**
   - Ensure `train_tradedata.csv` exists
   - Check Python dependencies are installed
   - Verify sufficient memory for model training

4. **Graph Visualization Issues**
   - Clear browser cache
   - Check JavaScript console for errors
   - Ensure internet connection for CDN resources

### Debug Mode
Run the application in debug mode for detailed error messages:
```bash
export FLASK_ENV=development
python app.py
```

## 📈 Performance Optimization

### Database Optimization
- Index frequently queried properties in Neo4j
- Use parameterized queries for better performance
- Implement connection pooling for high-traffic scenarios

### Application Optimization
- Enable caching for static data
- Implement pagination for large datasets
- Use background tasks for heavy computations

## 🔄 Future Enhancements

### Planned Features
- **Real-time Data Streaming**: Live data feeds from monitoring stations
- **Advanced ML Models**: Deep learning for complex predictions
- **Mobile App**: Native mobile application
- **API Docume# Water-quality-prediction-knowledge-graph
