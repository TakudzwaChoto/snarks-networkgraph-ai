import json
import pandas as pd
import threading #python定时器，进行实时监测数据
from py2neo import Graph, Node, Relationship

class neo4j:
    def __init__(self):
        self.g = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"), name='neo4j')

    def read_data(self):
        # 读取拓扑结构数据
        topo_data = pd.read_excel('河流拓扑结构.xlsx')
        # 读取氨氮统计数据
        nh4_data = pd.read_excel('河道氨氮统计数据--环境容量.xlsx')
        
        # 假设两表的连接字段为'Subbasin'或'RCH'
        merged_data = pd.merge(topo_data, nh4_data, left_on='Subbasin', right_on='RCH', how='left', suffixes=('_topo', '_nh4'))
        
        json_data = merged_data.to_dict(orient='records')
        return json_data

    def create_nodes(self, data):
        nodes = {}
        for d in data:
            node = Node('河流', objectid=d['Subbasin'], 
                        total_inflow=0, flow_out=d['FLOW_OUTcms'])
            self.g.create(node)
            nodes[d['Subbasin']] = node
        return nodes
    
    def create_rels(self, nodes, data):
        for d in data:
            from_node = nodes.get(d['FROM_NODE'])
            to_node = nodes.get(d['TO_NODE'])
            if from_node and to_node:
                
               
                # 更新节点属性
                from_node['flow_out'] = d['FLOW_OUTcms']*24*3600  #按照一天计算水流量为多少，立方米
                to_node['total_inflow'] +=  from_node['flow_out']
                
                # 推送更新到数据库
                self.g.push(from_node)
                self.g.push(to_node)
                
                # 创建关系
                relationship = Relationship(from_node, str(d['Subbasin']), to_node)
               
                relationship["AreaC"] = d["AreaC"]
                relationship["Len2"] = d["Len2"]
                relationship["Slo2"] = d["Slo2"]
                relationship["Wid2"] = d["Wid2"]
                relationship["Dep2"] = d["Dep2"]
                self.g.create(relationship)
    def create_stackholder_nodes(self, data):
        stackholder_nodes = {}
        for d in data:
            # 创建一个Stackholder节点
            stackholder_node = Node('Stackholder', id=d['RCH'], 
                                    #role=d['Role'],  # 假设Role列表示买家或卖家
                                    WEC=d['Cs'],
                                    RECORD=d['K'])
            self.g.create(stackholder_node)
            stackholder_nodes[d['RCH']] = stackholder_node
        return stackholder_nodes

    def create_stackholder_rels(self, stackholder_nodes, data):
        for d in data:
            buyer_node = stackholder_nodes.get(d['Buyer'])
            seller_node = stackholder_nodes.get(d['Seller'])
            if buyer_node and seller_node:
                # 创建交易关系
                transaction_rel = Relationship(buyer_node, "交易", seller_node)
                transaction_rel["Sold_wec"] = d["Sold_wec"] 
                self.g.create(transaction_rel)


    def create_rels_between_nodes_and_stackholder(self, nodes, stackholder_nodes):
    # 创建一个字典来存储已经创建的管理关系，以避免重复创建
        management_relations = {}

        for node_id, node in nodes.items():
            if node_id in stackholder_nodes:
                stackholder_node = stackholder_nodes[node_id]
                
                # 创建管理关系，如果还没有为这对节点创建过
                if node_id not in management_relations:
                    # 创建一个管理关系
                    management_relation = Relationship(stackholder_node, "MANAGES", node)
                    self.g.create(management_relation)
                    management_relations[node_id] = management_relation
            else:
                print(f"警告：没有找到与监测点 {node_id} 匹配的河流节点。")
        def check_nh3_concentration(self, threshold):  # 设定阈值
            print("开始检查氨氮浓度超过阈值的监测点...")
        
        # 执行Cypher查询
        query = """
        MATCH (m:监测点)
        WHERE m.nh3_concentration > $threshold
        RETURN m.id AS id, m.nh3_concentration AS nh3_concentration
        """
        
        # 尝试运行查询并捕获任何可能的异常
        try:
            results = self.g.run(query, threshold=threshold)
            print("查询执行成功.")
        except Exception as e:
          print(f"查询执行失败: {e}")
          return
        
        # 检查并打印结果
        result_count = 0
        for record in results:
            result_count += 1
            print(f"警告: 监测点 {record['id']} 的氨氮浓度 ({record['nh3_concentration']} mg/L) 超过阈值!")
        
        if result_count == 0:
           print("没有监测点的氨氮浓度超过阈值.")
        else:
           print(f"找到 {result_count} 个监测点的氨氮浓度超过阈值.")
          

    def export_nodes_and_rels(self):
        nodes_query = """
            MATCH (n)
            RETURN id(n) AS id, labels(n)[0] AS label, properties(n) AS properties 
        """
        relationships_query = """
            MATCH ()-[r]->()
            RETURN id(startNode(r)) AS source, id(endNode(r)) AS target, type(r) AS type,properties(r) AS properties
        """

        nodes = self.g.run(nodes_query).data()
        relationships = self.g.run(relationships_query).data()

        nodes_list = []
        for node in nodes:
            flow = node["properties"].get("flow", 0)
            total_inflow = node["properties"].get("total_inflow", 0)
            color = None
            if total_inflow > 100000:
                color = "red"
            elif 80000 < total_inflow <= 100000:
                color = "orange"
            elif 40000 < total_inflow <= 80000:
                color = "green"
            
            excluded_keys = ["flow", "total_inflow"]
            node_data = {
                "id": str(node["id"]),
                "label": node["label"],
                "flow": flow,
                "total_inflow": total_inflow,
                "color": color,
            }
            node_data.update({k: v for k, v in node["properties"].items() if k not in excluded_keys})
            nodes_list.append(node_data)

        relationships_list = [{"source": str(rel["source"]), "target": str(rel["target"]), "type": rel["type"],"properties": rel["properties"]} for rel in relationships]
        return {"nodes": nodes_list, "relationships": relationships_list}
    
    @staticmethod
    def run_periodically(interval, function, *args):
        timer = threading.Timer(interval, function, args=args)
        timer.daemon = True
        timer.start()

if __name__ == "__main__":
    
    handler = neo4j()
    data = handler.read_data()
    nodes = handler.create_nodes(data)
    handler.create_rels(nodes, data)  # 将data参数传递给create_rels

    
    stackholder_nodes = handler.create_stackholder_nodes(data)
    handler.create_stackholder_rels(stackholder_nodes, data)

    handler.create_rels_between_nodes_and_stackholder(nodes, stackholder_nodes)
    # handler.check_nh3_concentration(1.0)
    # neo4j.run_periodically(3600*4, handler.check_nh3_concentration(1.0))  # 每小时执行一次

    exported_data = handler.export_nodes_and_rels()
    with open('graph_data.json', 'w', encoding='utf-8') as f:
        json.dump(exported_data, f, ensure_ascii=False, indent=4)
    print("数据导出完成，保存为graph_data.json")