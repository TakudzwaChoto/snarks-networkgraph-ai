 #!/usr/bin/env python3
"""
Debug script to check Neo4j database status and data
"""

from py2neo import Graph
import pandas as pd

def test_neo4j_connection():
    """Test Neo4j connection and check data"""
    try:
        print("Testing Neo4j connection...")
        g = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"), name='neo4j')
        
        # Test basic connection
        test_query = "RETURN 1 as test"
        result = g.run(test_query).data()
        print("✅ Neo4j connection successful")
        
        # Count nodes
        count_query = "MATCH (n) RETURN count(n) as node_count"
        count_result = g.run(count_query).data()
        node_count = count_result[0]['node_count'] if count_result else 0
        print(f"📊 Total nodes in database: {node_count}")
        
        # Count relationships
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
        rel_count_result = g.run(rel_count_query).data()
        rel_count = rel_count_result[0]['rel_count'] if rel_count_result else 0
        print(f"🔗 Total relationships in database: {rel_count}")
        
        # Get node types
        node_types_query = "MATCH (n) RETURN labels(n)[0] as label, count(n) as count"
        node_types_result = g.run(node_types_query).data()
        print("📋 Node types:")
        for item in node_types_result:
            print(f"   - {item['label']}: {item['count']} nodes")
        
        # Get relationship types
        rel_types_query = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
        rel_types_result = g.run(rel_types_query).data()
        print("🔗 Relationship types:")
        for item in rel_types_result:
            print(f"   - {item['type']}: {item['count']} relationships")
        
        # Sample some nodes
        if node_count > 0:
            sample_query = "MATCH (n) RETURN labels(n)[0] as label, properties(n) LIMIT 3"
            sample_result = g.run(sample_query).data()
            print("📝 Sample nodes:")
            for item in sample_result:
                print(f"   - {item['label']}: {item['properties(n)']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("\n📁 Checking data files...")
    
    files_to_check = [
        '河流拓扑结构.xlsx',
        '河道氨氮统计数据--环境容量.xlsx',
        'train_tradedata.csv'
    ]
    
    for file in files_to_check:
        try:
            if file.endswith('.xlsx'):
                df = pd.read_excel(file)
                print(f"✅ {file}: {len(df)} rows, {len(df.columns)} columns")
            elif file.endswith('.csv'):
                df = pd.read_csv(file)
                print(f"✅ {file}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"❌ {file}: Error reading file - {e}")

def simulate_data_loading():
    """Simulate the data loading process"""
    print("\n🔄 Simulating data loading process...")
    
    try:
        # Read topology structure data
        print("Reading topology data...")
        topo_data = pd.read_excel('河流拓扑结构.xlsx')
        print(f"   Topology data: {len(topo_data)} rows")
        
        # Read ammonia nitrogen statistics
        print("Reading water quality data...")
        nh4_data = pd.read_excel('河道氨氮统计数据--环境容量.xlsx')
        print(f"   Water quality data: {len(nh4_data)} rows")
        
        # Merge data
        print("Merging data...")
        merged_data = pd.merge(topo_data, nh4_data, left_on='Subbasin', right_on='RCH', 
                             how='left', suffixes=('_topo', '_nh4'))
        print(f"   Merged data: {len(merged_data)} rows")
        
        # Check for required columns
        required_columns = ['Subbasin', 'FROM_NODE', 'TO_NODE', 'FLOW_OUTcms', 'RCH', 'Cs']
        missing_columns = [col for col in required_columns if col not in merged_data.columns]
        
        if missing_columns:
            print(f"   ⚠️  Missing columns: {missing_columns}")
        else:
            print("   ✅ All required columns present")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in data loading: {e}")
        return False

if __name__ == "__main__":
    print("🔍 River Management System - Database Debug Tool")
    print("=" * 50)
    
    # Check data files
    check_data_files()
    
    # Test Neo4j connection
    neo4j_ok = test_neo4j_connection()
    
    # Simulate data loading
    if neo4j_ok:
        simulate_data_loading()
    
    print("\n" + "=" * 50)
    print("🔍 Debug complete!")
    
    if neo4j_ok:
        print("💡 If the database is empty, try clicking 'Initialize Database' again")
        print("💡 If you see errors, check that Neo4j is running and accessible")
    else:
        print("💡 Please ensure Neo4j is running and the connection details are correct")