import numpy as np

# spatiotemporal graph
class ST_GRAPH:
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.nodes = [{} for i in range(batch_size)]
        self.edges = [{} for i in range(batch_size)]

    #构建ST图
    def Graph(self, source_batch):
        for sequence in range(self.batch_size):
            source_seq = source_batch[sequence]
            for framenum in range(self.seq_length):
                frame = source_seq[
                    framenum
                ]
                for ped in range(frame.shape[0]):
                    pedID = frame[ped, 0]
                    x = frame[ped, 1]
                    y = frame[ped, 2]
                    pos = (x, y)
                    node_type = frame[ped, 3]

                    if pedID not in self.nodes[sequence]:
                        node_id = pedID
                        node_pos_list = {}
                        node_pos_list[framenum] = pos
                        self.nodes[sequence][pedID] = ST_NODE(
                            node_type, node_id, node_pos_list
                        )

                    else:
                        self.nodes[sequence][pedID].addPosition(pos, framenum)

                        edge_id = (pedID, pedID)
                        pos_edge = (
                            self.nodes[sequence][pedID].getPosition(framenum - 1),
                            pos,
                        )

                        if edge_id not in self.edges[sequence]:
                            if int(node_type) == 3:
                                edge_type = "car/T"
                            else:
                                raise Exception("edge_type error")
                            edge_pos_list = {}
                            edge_pos_list[framenum] = pos_edge
                            self.edges[sequence][edge_id] = ST_EDGE(
                                edge_type, edge_id, edge_pos_list
                            )

                        else:
                            self.edges[sequence][edge_id].addPosition(
                                pos_edge, framenum
                            )
               # spatial edges
                for ped_in in range(frame.shape[0]):
                    for ped_out in range(ped_in + 1, frame.shape[0]):
                        pedID_in = frame[ped_in, 0]
                        pedID_out = frame[ped_out, 0]
                        pos_in = (frame[ped_in, 1], frame[ped_in, 2])
                        pos_out = (frame[ped_out, 1], frame[ped_out, 2])
                        pos = (pos_in, pos_out)
                        edge_id = (pedID_in, pedID_out)

                        if edge_id not in self.edges[sequence]:
                            edge_type = "edge/S"
                            edge_pos_list = {}
                            edge_pos_list[framenum] = pos
                            self.edges[sequence][edge_id] = ST_EDGE(
                                edge_type, edge_id, edge_pos_list
                            )
                        else:
                            self.edges[sequence][edge_id].addPosition(pos, framenum)

    def printGraph(self):

        for sequence in range(self.batch_size):
            nodes = self.nodes[sequence]
            edges = self.edges[sequence]

            print("Printing Nodes")
            print("===============================")
            for node in nodes.values():
                node.printNode()
                print("--------------")

            print("Printing Edges")
            print("===============================")
            for edge in edges.values():
                edge.printEdge()
                print("--------------")

    def getSequence(self):
        """
        Gets the sequence
        以 Numpy 数组的形式返回节点和边的序列
        序列主要是nodes, edges, nodesPresent, edgesPresent
        首先从中检索节点和边缘的字典self.nodes[0]和self.edges[0].
        """
        nodes = self.nodes[0]
        edges = self.edges[0]

        numNodes = len(nodes.keys())
        # print("********************* numNodes {}***********".format(numNodes))
        list_of_nodes = {}
        # 初始化数组，retNodes是一个3D numpy数组，其维度（seq_length，numNodes，2），代表每个时间步的节点位置。
        retNodes = np.zeros((self.seq_length, numNodes, 2))
        retEdges = np.zeros(
            (self.seq_length, numNodes * numNodes, 2)
        )  # Diagonal contains temporal edges
        # retNodePresent是一个2D数组列表，其元素是在每个时间步的节点位置。
        # retEdgePresent是一个2D数组列表，其元素是在每个时间步的边位置。
        retNodePresent = [[] for c in range(self.seq_length)]
        retEdgePresent = [[] for c in range(self.seq_length)]

        # 循环遍历节点字典的键，并将边缘添加到两个列表“retEdgePresent”和“retEdges”中，其中包含相应的信息
        # 并使用“node_pos，node_pos_list得到每个节点的信息
        for i, ped in enumerate(nodes.keys()):
            list_of_nodes[ped] = i
            pos_list = nodes[ped].node_pos_list
            for framenum in range(self.seq_length):
                if framenum in pos_list:
                    retNodePresent[framenum].append((i, nodes[ped].getType()))
                    retNodes[framenum, i, :] = list(pos_list[framenum])
                    # retNodes_type[framenum].append(nodes[ped].getType())

        for ped, ped_other in edges.keys():
            i, j = list_of_nodes[ped], list_of_nodes[ped_other]
            edge = edges[(ped, ped_other)]
            if ped == ped_other:
                # Temporal edge
                # 它会将边的信息（包括开始和结束节点索引及其类型）附加到“retEdgePresent”列表中，
                # 并将指向第二个节点的向量添加到“retEdges”数组中
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((i, j, edge.getType()))
                        retEdges[framenum, i * (numNodes) + j, :] = getVector(
                            edge.edge_pos_list[framenum]
                        )  # Gets the vector pointing from second element to first element
            else:
                # Spatial edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        #双向边
                        retEdgePresent[framenum].append((i, j, edge.getType()))
                        retEdgePresent[framenum].append((j, i, edge.getType()))
                        # the position returned is a tuple of tuples

                        retEdges[framenum, i * numNodes + j, :] = getVector(
                            edge.edge_pos_list[framenum]
                        )
                        retEdges[framenum, j * numNodes + i, :] = -np.copy(
                            retEdges[framenum, i * (numNodes) + j, :]
                        )

        return retNodes, retEdges, retNodePresent, retEdgePresent


class ST_NODE:
    def __init__(self, node_type, node_id, node_pos_list):
        self.node_type = node_type
        self.node_id = node_id
        self.node_pos_list = node_pos_list

    def getPosition(self, index_i):

        # 如果字典中不存在节点在给定时间步长的位置，
        # 则通过从字典中对键列表进行排序并查找小于的最后一个键来检索之前可用的最后一个位置
        if index_i not in self.node_pos_list:
            tmp_list = sorted(list(self.node_pos_list.keys()))#对字典的键进行顺序排序并将它们转换为列表
            last_index = [i for i in tmp_list if i < index_i][-1]
            return self.node_pos_list[last_index]#最后一个键
        # assert index in self.node_pos_list
        return self.node_pos_list[index_i]#返回 'self.node_pos_list 的值

    def getType(self):
        """
        Get node type
        """
        return self.node_type

    def getID(self):
        """
        Get node ID
        """
        return self.node_id

    def addPosition(self, pos, index):
        """
        Add position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        """
        assert index not in self.node_pos_list
        self.node_pos_list[index] = pos

    def printNode(self):
        """
        Print function for the node
        For debugging purposes
        """
        print(
            "Node type:",
            self.node_type,
            "with ID:",
            self.node_id,
            "with positions:",
            self.node_pos_list.values(),
            "at time-steps:",
            self.node_pos_list.keys(),
        )


class ST_EDGE:
    def __init__(self, edge_type, edge_id, edge_pos_list):
        self.edge_type = edge_type
        self.edge_id = edge_id
        self.edge_pos_list = edge_pos_list

    def getPositions(self, index):
        assert index in self.edge_pos_list
        return self.edge_pos_list[index]

    def getType(self):
        return self.edge_type

    def getID(self):
        return self.edge_id

    def addPosition(self, pos, index):
        assert index not in self.edge_pos_list
        self.edge_pos_list[index] = pos

    def printEdge(self):
        print(
            "Edge type:",
            self.edge_type,
            "between nodes:",
            self.edge_id,
            "with positions:",
            self.edge_pos_list.values(),
            "at time-steps:",
            self.edge_pos_list.keys(),
        )
