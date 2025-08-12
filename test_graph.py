 #!/usr/bin/env python3
"""
Test script to check the graph data API endpoint
"""

import requests
import json

def test_graph_api():
    """Test the graph data API endpoint"""
    try:
        print("🔍 Testing graph data API...")
        
        # Test the API endpoint
        response = requests.get('http://localhost:5000/api/graph-data')
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API call successful!")
            print(f"Nodes: {len(data.get('nodes', []))}")
            print(f"Relationships: {len(data.get('relationships', []))}")
            
            if len(data.get('nodes', [])) > 0:
                print("✅ Graph data available!")
                print(f"Sample node: {data['nodes'][0]}")
            else:
                print("❌ No nodes in graph data")
                
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def test_health_check():
    """Test the health check endpoint"""
    try:
        print("\n🔍 Testing health check...")
        
        response = requests.get('http://localhost:5000/health')
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check successful!")
            print(f"Status: {data.get('status')}")
            print(f"Neo4j: {data.get('services', {}).get('neo4j')}")
            print(f"ML Model: {data.get('services', {}).get('ml_model')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing health check: {e}")

if __name__ == "__main__":
    test_health_check()
    test_graph_api()