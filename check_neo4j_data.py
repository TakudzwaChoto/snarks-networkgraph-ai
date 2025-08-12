#!/usr/bin/env python3
"""
Check what data is currently in Neo4j database
"""

from py2neo import Graph
import json

def check_neo4j_data():
    """Check what data exists in Neo4j"""
    try:
        print("🔍 Connecting to Neo4j...")
        g = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"), name='neo4j')
        print("✅ Connected successfully!")
        
        print("\n" + "="*50)
        print("📊 DATABASE OVERVIEW")
        print("="*50)
        
        # Count all nodes
        node_count = g.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
        print(f"Total nodes: {node_count}")
        
        # Count all relationships
        rel_count = g.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']
        print(f"Total relationships: {rel_count}")
        
        if node_count == 0:
            print("\n❌ Database is empty! No data found.")
            print("💡 You need to either:")
            print("   1. Click 'Initialize Database' in the Flask app")
            print("   2. Add test data using /add-test-data")
            return
        
        print("\n" + "="*50)
        print("🏷️  NODE TYPES")
        print("="*50)
        
        # Get node types and counts
        node_types = g.run("""
            MATCH (n) 
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
        """).data()
        
        for item in node_types:
            print(f"{item['label']}: {item['count']} nodes")
        
        print("\n" + "="*50)
        print("🔗 RELATIONSHIP TYPES")
        print("="*50)
        
        # Get relationship types and counts
        rel_types = g.run("""
            MATCH ()-[r]->() 
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
        """).data()
        
        for item in rel_types:
            print(f"{item['type']}: {item['count']} relationships")
        
        print("\n" + "="*50)
        print("📝 SAMPLE DATA")
        print("="*50)
        
        # Show sample River nodes
        river_nodes = g.run("""
            MATCH (r:River) 
            RETURN r.objectid as id, r.total_inflow as inflow, r.flow_out as flow
            LIMIT 5
        """).data()
        
        if river_nodes:
            print("🌊 Sample River nodes:")
            for node in river_nodes:
                print(f"   ID: {node['id']}, Inflow: {node['inflow']}, Flow: {node['flow']}")
        else:
            print("🌊 No River nodes found")
        
        # Show sample MonitoringPoint nodes
        monitoring_nodes = g.run("""
            MATCH (m:MonitoringPoint) 
            RETURN m.id as id, m.nh3_concentration as nh3
            LIMIT 5
        """).data()
        
        if monitoring_nodes:
            print("\n📊 Sample MonitoringPoint nodes:")
            for node in monitoring_nodes:
                print(f"   ID: {node['id']}, NH3: {node['nh3']}")
        else:
            print("📊 No MonitoringPoint nodes found")
        
        # Show sample relationships
        relationships = g.run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] as from_type, type(r) as rel_type, labels(b)[0] as to_type
            LIMIT 5
        """).data()
        
        if relationships:
            print("\n🔗 Sample relationships:")
            for rel in relationships:
                print(f"   {rel['from_type']} -[{rel['rel_type']}]-> {rel['to_type']}")
        else:
            print("🔗 No relationships found")
        
        print("\n" + "="*50)
        print("🎯 GRAPH DATA FOR FLASK APP")
        print("="*50)
        
        # Test the exact query the Flask app uses
        try:
            nodes_query = """
                MATCH (n)
                RETURN id(n) AS id, labels(n)[0] AS label, properties(n) AS properties 
            """
            nodes = g.run(nodes_query).data()
            print(f"✅ Flask app can find {len(nodes)} nodes")
            
            relationships_query = """
                MATCH ()-[r]->()
                RETURN id(startNode(r)) AS source, id(endNode(r)) AS target, type(r) AS type
            """
            relationships = g.run(relationships_query).data()
            print(f"✅ Flask app can find {len(relationships)} relationships")
            
            if len(nodes) > 0:
                print("✅ Graph data should load successfully in Flask app!")
            else:
                print("❌ No data for Flask app to display")
                
        except Exception as e:
            print(f"❌ Error testing Flask app queries: {e}")
        
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        print("💡 Make sure Neo4j is running and the password is correct")

if __name__ == "__main__":
    check_neo4j_data()