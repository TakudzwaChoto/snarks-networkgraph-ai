from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import json
import pandas as pd
import numpy as np
import threading
import os
from datetime import datetime, timedelta
import time
from py2neo import Graph, Node, Relationship
import plotly.graph_objs as go
import plotly.utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

from deep_predictor import DeepPredictor
from zk_verifier import ZkVerifier
from audit_log import AuditLog
from config import Config
from ml_baselines import train_lgbm, train_xgb, predict as ml_predict
from federated import FederatedSimulator, FLConfig

try:
    from pydantic import BaseModel, ValidationError, field_validator
    PYD_AVAILABLE = True
except Exception:
    PYD_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

class RiverManagementSystem:
    def __init__(self):
        # Neo4j via config with retry/backoff
        cfg = Config()
        uri = cfg.NEO4J_URI
        user = cfg.NEO4J_USER
        pwd = cfg.NEO4J_PASSWORD
        self.neo4j_connected = False
        self.g = None
        for attempt in range(3):
            try:
                self.g = Graph(uri, auth=(user, pwd), name=cfg.NEO4J_DATABASE)
                # simple test
                self.g.run("RETURN 1 as ok").data()
                self.neo4j_connected = True
                break
            except Exception as e:
                print(f"Neo4j connection attempt {attempt+1} failed: {e}")
                time.sleep(1 + attempt)
        
        self.ml_model = None
        self.scaler = None
        self.audit = AuditLog()
        self.audit.append('system.start', {'neo4j_connected': self.neo4j_connected})
        self.load_or_train_model()

        # New deep predictor
        try:
            self.deep_predictor = DeepPredictor(train_csv='train_tradedata.csv', models_dir='models')
            print("Deep predictor ready")
            self.audit.append('model.deep.ready', {})
        except Exception as e:
            print(f"Failed to initialize deep predictor: {e}")
            self.audit.append('model.deep.error', {'error': str(e)})
            self.deep_predictor = None

        # zkSNARK verifier
        try:
            self.zk = ZkVerifier()
            print("zk verifier ready")
            self.audit.append('zk.ready', {'available': self.zk.is_available(), 'configured': self.zk.is_configured()})
        except Exception as e:
            print(f"Failed to init zk verifier: {e}")
            self.audit.append('zk.error', {'error': str(e)})
            self.zk = None

    def load_or_train_model(self):
        """Load existing model or train a new one"""
        model_path = 'river_quality_model.pkl'
        scaler_path = 'river_quality_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.ml_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print("Loaded existing ML model")
            except:
                self.train_model()
        else:
            self.train_model()

    def train_model(self):
        """Train machine learning model for water quality prediction"""
        try:
            # Load training data
            train_data = pd.read_csv('train_tradedata.csv')
            
            # Create features for ML model
            features = []
            targets = []
            
            for _, row in train_data.iterrows():
                buyer = row['Buyer']
                seller = row['Seller']
                sold_wec = row['Sold_wec']
                
                # Create features based on buyer and seller characteristics
                features.append([
                    buyer, seller,  # Basic identifiers
                    abs(buyer - seller),  # Distance between segments
                    buyer % 10, seller % 10,  # Some derived features
                    (buyer + seller) % 20
                ])
                targets.append(sold_wec)
            
            X = np.array(features)
            y = np.array(targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Save model
            joblib.dump(self.ml_model, 'river_quality_model.pkl')
            joblib.dump(self.scaler, 'river_quality_scaler.pkl')
            
            print("ML model trained and saved successfully")
            
        except Exception as e:
            print(f"Error training model: {e}")

    def predict_water_quality_trade(self, buyer_segment, seller_segment):
        """Predict water quality trading amount between segments"""
        # Prefer deep predictor if available
        if getattr(self, 'deep_predictor', None) is not None:
            try:
                return self.deep_predictor.predict(int(buyer_segment), int(seller_segment))
            except Exception:
                pass

        if self.ml_model is None:
            return 0
        
        try:
            features = np.array([[
                buyer_segment, seller_segment,
                abs(buyer_segment - seller_segment),
                buyer_segment % 10, seller_segment % 10,
                (buyer_segment + seller_segment) % 20
            ]])
            
            features_scaled = self.scaler.transform(features)
            prediction = self.ml_model.predict(features_scaled)[0]
            return max(0, round(prediction, 2))
        except:
            return 0

    def read_data(self):
        """Read river topology and water quality data"""
        try:
            # Read topology structure data
            topo_data = pd.read_excel('河流拓扑结构.xlsx')
            # Read ammonia nitrogen statistics
            nh4_data = pd.read_excel('河道氨氮统计数据--环境容量.xlsx')
            
            # Merge data
            merged_data = pd.merge(topo_data, nh4_data, left_on='Subbasin', right_on='RCH', 
                                 how='left', suffixes=('_topo', '_nh4'))
            
            return merged_data.to_dict(orient='records')
        except Exception as e:
            print(f"Error reading data: {e}")
            return []

    def create_nodes(self, data):
        """Create river nodes in Neo4j"""
        if not self.neo4j_connected:
            return {}
        
        nodes = {}
        for d in data:
            try:
                node = Node('River', 
                           objectid=d.get('Subbasin', ''),
                           total_inflow=0, 
                           flow_out=d.get('FLOW_OUTcms', 0),
                           area=d.get('AreaC', 0),
                           length=d.get('Len2', 0),
                           slope=d.get('Slo2', 0),
                           width=d.get('Wid2', 0),
                           depth=d.get('Dep2', 0))
                self.g.create(node)
                nodes[d.get('Subbasin', '')] = node
            except Exception as e:
                print(f"Error creating node: {e}")
        return nodes

    def create_relationships(self, nodes, data):
        """Create relationships between river nodes"""
        if not self.neo4j_connected:
            return
        
        for d in data:
            try:
                from_node = nodes.get(d.get('FROM_NODE'))
                to_node = nodes.get(d.get('TO_NODE'))
                
                if from_node and to_node:
                    # Update node properties
                    flow_out = d.get('FLOW_OUTcms', 0) * 24 * 3600  # Daily flow in cubic meters
                    from_node['flow_out'] = flow_out
                    to_node['total_inflow'] = to_node.get('total_inflow', 0) + flow_out
                    
                    # Push updates to database
                    self.g.push(from_node)
                    self.g.push(to_node)
                    
                    # Create relationship
                    relationship = Relationship(from_node, 'FLOWS_TO', to_node)
                    relationship["AreaC"] = d.get("AreaC", 0)
                    relationship["Len2"] = d.get("Len2", 0)
                    relationship["Slo2"] = d.get("Slo2", 0)
                    relationship["Wid2"] = d.get("Wid2", 0)
                    relationship["Dep2"] = d.get("Dep2", 0)
                    self.g.create(relationship)
            except Exception as e:
                print(f"Error creating relationship: {e}")

    def create_monitoring_nodes(self, data):
        """Create monitoring point nodes"""
        if not self.neo4j_connected:
            return {}
        
        monitoring_nodes = {}
        for d in data:
            try:
                monitoring_node = Node('MonitoringPoint', 
                                     id=d.get('RCH', ''),
                                     wec=d.get('Cs', 0),
                                     record=d.get('K', 0),
                                     nh3_concentration=d.get('Cs', 0))  # Using Cs as NH3 concentration
                self.g.create(monitoring_node)
                monitoring_nodes[d.get('RCH', '')] = monitoring_node
            except Exception as e:
                print(f"Error creating monitoring node: {e}")
        return monitoring_nodes

    def create_monitoring_relationships(self, nodes, monitoring_nodes):
        """Create relationships between river nodes and monitoring points"""
        if not self.neo4j_connected:
            return
        
        for node_id, node in nodes.items():
            if node_id in monitoring_nodes:
                try:
                    monitoring_node = monitoring_nodes[node_id]
                    management_relation = Relationship(monitoring_node, "MONITORS", node)
                    self.g.create(management_relation)
                except Exception as e:
                    print(f"Error creating monitoring relationship: {e}")

    def get_graph_data(self):
        """Export graph data for visualization"""
        if not self.neo4j_connected:
             print("Neo4j not connected - returning empty graph data")
             return {"nodes": [], "relationships": []}
        
        try:
            print("Attempting to query Neo4j for graph data...")
            
            # First, let's check if there are any nodes at all
            count_query = "MATCH (n) RETURN count(n) as node_count"
            count_result = self.g.run(count_query).data()
            node_count = count_result[0]['node_count'] if count_result else 0
            print(f"Total nodes in database: {node_count}")
            
            if node_count == 0:
                print("No nodes found in database - returning empty graph")
                return {"nodes": [], "relationships": []}
            
            nodes_query = """
                MATCH (n)
                RETURN id(n) AS id, labels(n)[0] AS label, properties(n) AS properties 
            """
            relationships_query = """
                MATCH ()-[r]->()
                RETURN id(startNode(r)) AS source, id(endNode(r)) AS target, type(r) AS type
            """

            print("Executing nodes query...")
            nodes = self.g.run(nodes_query).data()
            print(f"Found {len(nodes)} nodes")
            
            print("Executing relationships query...")
            relationships = self.g.run(relationships_query).data()
            print(f"Found {len(relationships)} relationships")

            nodes_list = []
            for node in nodes:
                try:
                    properties = node["properties"]
                    total_inflow = properties.get("total_inflow", 0)
                    
                    # Determine color based on inflow
                    if total_inflow > 100000:
                        color = "#ff4444"  # Red
                    elif 80000 < total_inflow <= 100000:
                        color = "#ff8800"  # Orange
                    elif 40000 < total_inflow <= 80000:
                        color = "#44ff44"  # Green
                    else:
                        color = "#4444ff"  # Blue
                    
                    node_data = {
                        "id": str(node["id"]),
                        "label": node["label"],
                        "color": color,
                        "size": min(20, max(5, total_inflow / 10000)),  # Size based on inflow
                        "properties": properties
                    }
                    nodes_list.append(node_data)
                except Exception as e:
                    print(f"Error processing node {node}: {e}")
                    continue

            relationships_list = []
            for rel in relationships:
                try:
                    rel_data = {
                        "source": str(rel["source"]), 
                        "target": str(rel["target"]), 
                        "type": rel["type"]
                    }
                    relationships_list.append(rel_data)
                except Exception as e:
                    print(f"Error processing relationship {rel}: {e}")
                    continue
            
            result = {"nodes": nodes_list, "relationships": relationships_list}
            print(f"Returning graph data: {len(nodes_list)} nodes, {len(relationships_list)} relationships")
            return result
            
        except Exception as e:
            print(f"Error getting graph data: {e}")
            import traceback
            traceback.print_exc()
            return {"nodes": [], "relationships": []}

    def check_water_quality_alerts(self, threshold=1.0):
        """Check for water quality alerts"""
        if not self.neo4j_connected:
            return []
        
        try:
            query = """
            MATCH (m:MonitoringPoint)
            WHERE m.nh3_concentration > $threshold
            RETURN m.id AS id, m.nh3_concentration AS nh3_concentration
            """
            
            results = self.g.run(query, threshold=threshold)
            alerts = []
            
            for record in results:
                alerts.append({
                    'id': record['id'],
                    'nh3_concentration': record['nh3_concentration'],
                    'threshold': threshold
                })
            
            return alerts
        except Exception as e:
            print(f"Error checking water quality alerts: {e}")
            return []

    def get_water_quality_stats(self):
        """Get water quality statistics"""
        if not self.neo4j_connected:
            return {}
        
        try:
            query = """
            MATCH (m:MonitoringPoint)
            RETURN 
                avg(m.nh3_concentration) as avg_nh3,
                max(m.nh3_concentration) as max_nh3,
                min(m.nh3_concentration) as min_nh3,
                count(m) as total_points
            """
            
            result = self.g.run(query).data()
            if result:
                return result[0]
            return {}
        except Exception as e:
            print(f"Error getting water quality stats: {e}")
            return {}

# Initialize the system
river_system = RiverManagementSystem()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'neo4j': river_system.neo4j_connected,
            'ml_model': river_system.ml_model is not None,
            'flask': True
        }
    }
    
    if not river_system.neo4j_connected:
        status['status'] = 'degraded'
        status['message'] = 'Neo4j connection failed'
    
    return jsonify(status)

@app.route('/api/graph-data')
def get_graph_data():
    """API endpoint to get graph data"""
    try:
        data = river_system.get_graph_data()
        return jsonify(data)
    except Exception as e:
        print(f"Error in get_graph_data: {e}")
        return jsonify({"nodes": [], "relationships": []})

@app.route('/api/water-quality-alerts')
def get_water_quality_alerts():
    """API endpoint to get water quality alerts"""
    threshold = request.args.get('threshold', 1.0, type=float)
    alerts = river_system.check_water_quality_alerts(threshold)
    return jsonify(alerts)

@app.route('/api/water-quality-stats')
def get_water_quality_stats():
    """API endpoint to get water quality statistics"""
    stats = river_system.get_water_quality_stats()
    return jsonify(stats)

@app.route('/api/predict', methods=['POST'])
def predict_unified():
    data = request.get_json() or {}
    buyer = data.get('buyer')
    seller = data.get('seller')
    model = data.get('model', 'deep')

    # Validate
    if PYD_AVAILABLE:
        class Req(BaseModel):
            buyer: int
            seller: int
            model: str = 'deep'
            @field_validator('buyer','seller')
            @classmethod
            def non_negative(cls, v):
                if v is None or int(v) < 0:
                    raise ValueError('must be non-negative int')
                return int(v)
        try:
            req = Req(buyer=buyer, seller=seller, model=model)
            buyer, seller, model = req.buyer, req.seller, req.model
        except ValidationError as e:
            return jsonify({'status':'error','errors': json.loads(e.json())}), 400
    else:
        if buyer is None or seller is None:
            return jsonify({'status':'error','message':'buyer and seller required'}), 400
        buyer = int(buyer); seller = int(seller)

    if model == 'deep':
        pred = river_system.predict_water_quality_trade(buyer, seller)
        try:
            river_system.audit.append('predict.deep', {'buyer': buyer, 'seller': seller, 'prediction': pred})
        except Exception:
            pass
        return jsonify({'prediction': pred, 'model': 'deep'})
    elif model in ('lgbm','xgb'):
        topo = Config.TOPOLOGY_FILE if hasattr(Config, 'TOPOLOGY_FILE') else '河流拓扑结构.xlsx'
        res = ml_predict(model, buyer, seller, topo)
        if 'error' in res:
            try:
                river_system.audit.append('predict.baseline.error', {'model': model, 'buyer': buyer, 'seller': seller, 'error': res['error']})
            except Exception:
                pass
            return jsonify({'status':'error','message':res['error']}), 400
        res['model'] = model
        try:
            river_system.audit.append('predict.baseline', {'model': model, 'buyer': buyer, 'seller': seller, 'prediction': res.get('prediction')})
        except Exception:
            pass
        return jsonify(res)
    else:
        return jsonify({'status':'error','message':'unknown model'}), 400

@app.route('/api/train-ml', methods=['POST'])
def train_ml():
    payload = request.get_json() or {}
    model = payload.get('model','lgbm')
    topo = Config.TOPOLOGY_FILE if hasattr(Config, 'TOPOLOGY_FILE') else '河流拓扑结构.xlsx'
    if model == 'lgbm':
        out = train_lgbm('train_tradedata.csv', topology_xlsx=topo)
    elif model == 'xgb':
        out = train_xgb('train_tradedata.csv', topology_xlsx=topo)
    else:
        return jsonify({'status':'error','message':'unknown model'}), 400
    if 'error' in out:
        try:
            river_system.audit.append('train.baseline.error', {'model': model, 'error': out['error']})
        except Exception:
            pass
        return jsonify({'status':'error','message':out['error']}), 400
    try:
        river_system.audit.append('train.baseline.ok', {'model': model, 'metrics': out.get('metrics'), 'used_graph': out.get('used_graph')})
    except Exception:
        pass
    return jsonify({'status':'ok','result': out})

@app.route('/api/train-deep', methods=['POST'])
def train_deep():
    """Trigger deep model training (retrain)."""
    try:
        if river_system.deep_predictor is None:
            river_system.deep_predictor = DeepPredictor(train_csv='train_tradedata.csv', models_dir='models')
        metrics = river_system.deep_predictor.train()
        try:
            river_system.audit.append('train.deep.ok', {'metrics': metrics})
        except Exception:
            pass
        return jsonify({'status': 'ok', 'metrics': metrics})
    except Exception as e:
        try:
            river_system.audit.append('train.deep.error', {'error': str(e)})
        except Exception:
            pass
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/evaluate-deep')
def evaluate_deep():
    try:
        if river_system.deep_predictor is None:
            return jsonify({'status': 'error', 'message': 'deep predictor not initialized'}), 400
        metrics = river_system.deep_predictor.evaluate()
        return jsonify({'status': 'ok', 'metrics': metrics})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/feature-importance')
def feature_importance():
    try:
        if river_system.deep_predictor is None:
            return jsonify({'status': 'error', 'message': 'deep predictor not initialized'}), 400
        result = river_system.deep_predictor.feature_importance()
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/zk/verify', methods=['POST'])
def zk_verify():
    """Verify a submitted zkSNARK proof and public signals.
    Expected JSON body: { "proof": {...}, "publicSignals": [...] }
    """
    try:
        if river_system.zk is None:
            return jsonify({'status': 'error', 'message': 'zk verifier unavailable'}), 400
        payload = request.get_json(force=True)
        proof = payload.get('proof')
        public_signals = payload.get('publicSignals')
        if proof is None or public_signals is None:
            return jsonify({'status': 'error', 'message': 'missing proof or publicSignals'}), 400
        ok = river_system.zk.verify(proof, public_signals)
        try:
            river_system.audit.append('zk.verify', {'verified': bool(ok)})
        except Exception:
            pass
        return jsonify({'status': 'ok', 'verified': ok})
    except Exception as e:
        try:
            river_system.audit.append('zk.verify.error', {'error': str(e)})
        except Exception:
            pass
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/zk/metrics')
def zk_metrics():
    try:
        if river_system.zk is None:
            return jsonify({'status': 'error', 'message': 'zk verifier unavailable'}), 400
        stats = river_system.zk.stats()
        return jsonify({'status': 'ok', 'stats': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/fl/init', methods=['POST'])
def fl_init():
    try:
        payload = request.get_json() or {}
        cfg = FLConfig(
            n_clients=int(payload.get('n_clients', 5)),
            client_fraction=float(payload.get('client_fraction', 1.0)),
            local_epochs=int(payload.get('local_epochs', 1)),
            batch_size=int(payload.get('batch_size', 64)),
            lr=float(payload.get('lr', 1e-2)),
            clip_norm=float(payload.get('clip_norm', 1.0)),
            seed=int(payload.get('seed', 42)),
        )
        river_system.fl = FederatedSimulator(river_system.audit)
        res = river_system.fl.init(cfg)
        return jsonify(res)
    except Exception as e:
        return jsonify({'status':'error','message': str(e)}), 400


@app.route('/api/fl/round', methods=['POST'])
def fl_round():
    try:
        if getattr(river_system, 'fl', None) is None:
            return jsonify({'status':'error','message':'fl not initialized'}), 400
        res = river_system.fl.run_round()
        return jsonify(res)
    except Exception as e:
        return jsonify({'status':'error','message': str(e)}), 400


@app.route('/api/fl/status')
def fl_status():
    try:
        if getattr(river_system, 'fl', None) is None:
            return jsonify({'status':'error','message':'fl not initialized'}), 400
        return jsonify(river_system.fl.status())
    except Exception as e:
        return jsonify({'status':'error','message': str(e)}), 400


@app.route('/api/fl/predict')
def fl_predict():
    try:
        if getattr(river_system, 'fl', None) is None:
            return jsonify({'status':'error','message':'fl not initialized'}), 400
        buyer = int(request.args.get('buyer', 1))
        seller = int(request.args.get('seller', 2))
        yhat = river_system.fl.predict(buyer, seller)
        river_system.audit.append('fl.predict', {'buyer': buyer, 'seller': seller, 'prediction': yhat})
        return jsonify({'prediction': yhat})
    except Exception as e:
        return jsonify({'status':'error','message': str(e)}), 400

@app.route('/initialize-database')
def initialize_database():
    """Initialize the Neo4j database with river data"""
    try:
        data = river_system.read_data()
        if data:
            nodes = river_system.create_nodes(data)
            river_system.create_relationships(nodes, data)
            monitoring_nodes = river_system.create_monitoring_nodes(data)
            river_system.create_monitoring_relationships(nodes, monitoring_nodes)
            try:
                river_system.audit.append('db.init.ok', {'nodes': len(nodes), 'monitoring_points': len(monitoring_nodes)})
            except Exception:
                pass
            flash('Database initialized successfully!', 'success')
        else:
            try:
                river_system.audit.append('db.init.empty', {})
            except Exception:
                pass
            flash('No data found to initialize database', 'error')
    except Exception as e:
        try:
            river_system.audit.append('db.init.error', {'error': str(e)})
        except Exception:
            pass
        flash(f'Error initializing database: {e}', 'error')
    
    return redirect(url_for('index'))

@app.route('/add-test-data')
def add_test_data():
    """Add test data to the database for debugging"""
    try:
        if not river_system.neo4j_connected:
            flash('Neo4j not connected', 'error')
            return redirect(url_for('index'))
        
        # Create some test river nodes
        test_nodes = []
        for i in range(1, 6):
            node = Node('River', 
                       objectid=f'R{i}',
                       total_inflow=i * 10000,
                       flow_out=i * 5000,
                       area=i * 200,
                       length=i * 100,
                       slope=i * 0.1,
                       width=i * 10,
                       depth=i * 2)
            river_system.g.create(node)
            test_nodes.append(node)
        
        # Create relationships between nodes
        for i in range(len(test_nodes) - 1):
            rel = Relationship(test_nodes[i], 'FLOWS_TO', test_nodes[i + 1])
            river_system.g.create(rel)
        
        # Create some monitoring points
        for i in range(1, 4):
            monitoring_node = Node('MonitoringPoint',
                                 id=f'M{i}',
                                 wec=i * 0.5,
                                 record=i,
                                 nh3_concentration=i * 0.3)
            river_system.g.create(monitoring_node)
            
            # Connect to river nodes
            if i <= len(test_nodes):
                rel = Relationship(monitoring_node, 'MONITORS', test_nodes[i-1])
                river_system.g.create(rel)
        
        try:
            river_system.audit.append('db.testdata.ok', {'river_nodes': len(test_nodes), 'monitoring_points': 3})
        except Exception:
            pass
        flash('Test data added successfully!', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        try:
            river_system.audit.append('db.testdata.error', {'error': str(e)})
        except Exception:
            pass
        flash(f'Error adding test data: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/test-database')
def test_database():
    """Test endpoint to check database status and data"""
    try:
        if not river_system.neo4j_connected:
            return jsonify({
                'status': 'error',
                'message': 'Neo4j not connected'
            })
        
        # Test basic connection
        test_query = "RETURN 1 as test"
        result = river_system.g.run(test_query).data()
        
        # Count nodes
        count_query = "MATCH (n) RETURN count(n) as node_count"
        count_result = river_system.g.run(count_query).data()
        node_count = count_result[0]['node_count'] if count_result else 0
        
        # Count relationships
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
        rel_count_result = river_system.g.run(rel_count_query).data()
        rel_count = rel_count_result[0]['rel_count'] if rel_count_result else 0
        
        # Get sample nodes
        sample_query = "MATCH (n) RETURN labels(n)[0] as label, count(n) as count LIMIT 5"
        sample_result = river_system.g.run(sample_query).data()
        
        return jsonify({
            'status': 'success',
            'connection': 'OK',
            'node_count': node_count,
            'relationship_count': rel_count,
            'node_types': sample_result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
@app.route('/debug-graph')
def debug_graph():
    """Debug page for testing graph visualization"""
    return render_template('debug_graph.html')
 
@app.route('/dashboard')
def dashboard():
    """Dashboard with charts and analytics"""
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)