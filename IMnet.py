from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
import math
import heapq
import random
import time


def logistic_normalization(file_in_path, file_out_path):
    file_in = open(file_in_path, "r")
    file_out = open(file_out_path, "w")
    string = file_in.readline()
    while (string != None) & (string != ""):
        tmp = string.split()
        tmp0 = ((1 / (1 + math.exp(-int(tmp[0])))) - 0.5) * 2
        tmp1 = ((1 / (1 + math.exp(-float(tmp[1])))) - 0.5) * 2
        file_out.write(str(tmp0) + " " + str(tmp1) + "\n")
        string = file_in.readline()
    file_in.close()
    file_out.close()


def degree_max_top_k(k, node_list):
    is_visited = [0 for i in range(len(node_list))]
    topk = []
    for i in range(k):
        max_degree = -1
        index = -1
        for j in range(len(node_list)):
            if is_visited[j] == 1:
                continue
            else:
                if max_degree < node_list[j][0]:
                    max_degree = node_list[j][0]
                    index = j
        topk.append(index)
        is_visited[index] = 1
    return topk


def min_max_normalization(file_in_path, file_out_path):
    file_in = open(file_in_path, "r")
    file_out = open(file_out_path, "w")
    string = file_in.readline()
    max_degree = -1
    max_sum_weight = -1
    min_degree = float("inf")
    min_sum_weight = float("inf")
    while (string != None) & (string != ""):
        tmp = string.split()
        if max_degree < int(tmp[0]):
            max_degree = int(tmp[0])
        elif min_degree > int(tmp[0]):
            min_degree = int(tmp[0])

        if max_sum_weight < float(tmp[1]):
            max_sum_weight = float(tmp[1])
        elif min_sum_weight > float(tmp[1]):
            min_sum_weight = float(tmp[1])
        string = file_in.readline()
    file_in.close()
    file_out.close()
    file_in = open(file_in_path, "r")
    file_out = open(file_out_path, "w")
    string = file_in.readline()
    while (string != None) & (string != ""):
        tmp = string.split()
        tmp0 = (int(tmp[0]) - min_degree)/(max_degree - min_degree)
        tmp1 = (float(tmp[1]) - min_sum_weight)/(max_sum_weight - min_sum_weight)
        file_out.write(str(tmp0) + " " + str(tmp1) + "\n")
        string = file_in.readline()
    file_in.close()
    file_out.close()


class EdgeTuple:
    start_node = -1
    end_node = -1
    info = []
    feature_num = 0
    edge_cmp_feature_choice = 0

    def __init__(self, start_node, end_node, feature_num, cmp_choice = 0):
        if start_node < 0 | end_node < 0:
            print ("input wrong parameter:start_node and end_node's value can't less than zero !")
        else:
            self.start_node = start_node
            self.end_node = end_node
            self.feature_num = feature_num
            self.info = [[]for i in range(self.feature_num)]
            self.edge_cmp_feature_choice = cmp_choice

    def set_feature(self, feature_loc, feature_value):
        if feature_loc >= self.feature_num:
            print ("wrong input of parameter feature_loc")
        else:
            self.info[feature_loc] = feature_value

    def __lt__(self, other):
        return self.info[self.edge_cmp_feature_choice] < other.info[other.edge_cmp_feature_choice]


class FileIOForGraph:
    read_path = ''
    write_path = ''
    __file_in = None
    __file_out = None
    __Num = 0

    def __init__(self, read_path, Num):
        self.read_path = read_path
        self.__Num = Num


# assume that the edges is sorted in start_node's order,and any vertex has 2 list,
# one is out_edge_list while another is in_edge_list
# the net is stored in adj_list
# and every tuple is a triangle (start_node, end_node, weight)

    def read_net_graph(self):
        self.__file_in = open(self.read_path, "r")
        a = [[[], []] for i in range(self.__Num)]
        str_tmp = self.__file_in.readline()
        while str_tmp != "":
            b = str_tmp.split()
            start_node = int(b[0])
            end_node = int(b[1])
            a[start_node][0].append([end_node, 0])
            a[end_node][1].append([start_node, 0])

            str_tmp = self.__file_in.readline()
        self.__file_in.close()
        return a


class Net:
    # the adjacent list for every node
    # assume that the edges is sorted in start_node's order,and any vertex has 2 list,
    # one is out_edge_list while another is in_edge_list
    # the net is stored in adj_list
    # and every tuple is a triangle (start_node, end_node, weight)
    __adjacent = None
    # file io class
    __file_io = None
    # network influence propagation model, default as weighted_cascade_model
    propagation_model = ""
    # the number of nodes
    node_num = 0
    # the list of nodes which stores the information of nodes , such as location , age , etc.
    node_list = None
    mioa = None
    miia = None

    def __init__(self, read_path, propagation_model, node_num=0):
        self.__file_io = FileIOForGraph(read_path, node_num)
        self.__adjacent = self.__file_io.read_net_graph()
        self.propagation_model = propagation_model
        self.node_num = node_num
        self.node_list = [[] for i in range(node_num)]
        if propagation_model == "MIA":
            self.__weight_cascade()

    def __weight_cascade(self):
        for i in range(len(self.__adjacent)):
            self.__adjacent[i][0] = []

        for i in range(len(self.__adjacent)):
            length = len(self.__adjacent[i][1])
            log_weight = 0
            if length != 0:
                weight = 1 / length
                log_weight = 0 - math.log10(weight)
            else:
                weight = 0
            for j in range(length):
                self.__adjacent[i][1][j][1] = weight
                self.__adjacent[i][1][j].append(log_weight)
                start_node = self.__adjacent[i][1][j][0]
                end_node = i
                self.__adjacent[start_node][0].append([end_node, weight, log_weight])

# by using Dijstra algorithm, this method can find out every node to others' shortest path,
# and then maintain two list which are called MIIA and MIOA for every node
# MIIA(v)(maximum influence in arborescence) stores the nodes which can reach v through the MIP(u, v)
# MIOA(v)(maximum influence in arborescence) stores the nodes which can be reached by v through the MIP(v, u)
# MIP(x, y) represent a path which means x influence y with the maximum probability

    def gen_mia_model(self, is_edge_weighted, mioa_path="D:/mioa.txt", miia_path="D:/miia.txt"):
        # calculate -log() for every edge's weight, and append the -log(weight) to the tuple of edges.
        # for example: (start_node, end_node, weight, -log(weight))
        if is_edge_weighted == 1:
            #
            return
        else:
            self.miia = []
            self.mioa = []
            for i in range(self.node_num):
                self.mioa.append(self.__dijstra(0, "log", i, 0.1))
            for i in range(self.node_num):
                self.miia.append(self.__dijstra(1, "log", i, 0.1))
            miia_file = open(miia_path, "w")
            mioa_file = open(mioa_path, "w")
            for i in range(self.node_num):
                for j in range(len(self.miia[i])):
                    miia_file.write(str(self.miia[i][j][0]) + "," + str(self.miia[i][j][1]) + "," +
                                    str(self.miia[i][j][2]) + "," + str(self.miia[i][j][3]) + " ")
                miia_file.write("\n")
                for j in range(len(self.mioa[i])):
                    mioa_file.write(str(self.mioa[i][j][0]) + "," + str(self.mioa[i][j][1]) + "," +
                                    str(self.mioa[i][j][2]) + "," + str(self.mioa[i][j][3]) + " ")
                mioa_file.write("\n")
            mioa_file.close()
            miia_file.close()

    # the return parameter of this method is the list of MIOA of every nodes
    # here the parameter weight_choice represents the type of weight
    def __dijstra(self, iora, weight_choice="log", start_node=0, threshold=0.5):
        print (start_node)
        mioa_node_list = []
        log_threshold = 0 - math.log10(threshold)
        if weight_choice == "weight":
            #
            return
        elif weight_choice == "log":
            if self.propagation_model != "MIA":
                return None
            # build a list storing the length of path from start_node to others
            # first element represents the length of the path
            # second element represents the node number
            # third element represents the times of relaxing operation
            # 更改了开始的决定，决定mioa存边表吧
            length_to_start = [[float("inf"), i, float("inf")] for i in range(self.node_num)]
            path_to_start = [[-1, float("inf")] for i in range(self.node_num)]
            fixed_length_to_start = []
            # the list below  represents
            # the set S which contain all the elements with value 1
            # and the set V-S which contain all the elements with calue 0
            is_visited = [0 for i in range(self.node_num)]
            for i in range(len(length_to_start)):
                fixed_length_to_start.append(length_to_start[i])
            # set length of  the start_node self = 0
            for i in range(len(length_to_start)):
                if length_to_start[i][1] == start_node:
                    length_to_start[i][0] = 0

            # build a heap
            heapq.heapify(length_to_start)
            top = heapq.heappop(length_to_start)
            # need to optimize the code, it was so slow......
            is_visited[top[1]] = 1
            top[2] = 0

            while top[0] != float("inf"):
                # besides, we also needn't to worry that the rest part will be written into the mioa of start node
                # because of the property of the Dijstra algorithm,
                # the node chosen in step i had found the minimum path
                # if current choosing node had larger value than log threshold
                # than we can stop the process immediately
                if (top[0] > log_threshold) | (len(length_to_start) == 0):
                    break

                for i in range(len(fixed_length_to_start)):
                    if is_visited[i] == 0:
                        self.__relax(top[1], i, fixed_length_to_start, path_to_start, iora)
                    else:
                        continue
                heapq.heapify(length_to_start)
                top = heapq.heappop(length_to_start)
                is_visited[top[1]] = 1

            # for i in range(len(fixed_length_to_start)):
            #     if fixed_length_to_start[i][0] < log_threshold:
            #         mip = 10**(-fixed_length_to_start[i][0])
            for i in range(len(fixed_length_to_start)):
                if i == start_node:
                    continue
                if fixed_length_to_start[i][0] < log_threshold:
                    if iora == 0:
                        mioa_node_list.append([path_to_start[i][0], i, path_to_start[i][1], fixed_length_to_start[i][2]])
                    else:
                        mioa_node_list.append([i, path_to_start[i][0], path_to_start[i][1], fixed_length_to_start[i][2]])
            return sorted(mioa_node_list, key=lambda mioa_node_list: mioa_node_list[3])
        else:
            # execute relaxing operation about edge(u->v)
            return None

    def __relax(self, u, v, length_to_start, path_to_start, ioro=0):
        cost_u_v = float("inf")
        weight_u_v = float("inf")
        for i in range(len(self.__adjacent[u][ioro])):
            if self.__adjacent[u][ioro][i][0] == v:
                cost_u_v = self.__adjacent[u][ioro][i][2]
                weight_u_v = self.__adjacent[u][ioro][i][1]
                break

        if length_to_start[v][0] > length_to_start[u][0] + cost_u_v:
            length_to_start[v][0] = length_to_start[u][0] + cost_u_v
            length_to_start[v][2] = length_to_start[u][2] + 1
            path_to_start[v][0] = u
            path_to_start[v][1] = weight_u_v

    def mia_greedy(self, seed_num, candidates):
        # 需要维护一个种子集合的mioa网络，节省不必要的节点的ap计算
        if (self.miia == None) | (self.mioa == None):
            print ("No miia or mioa net!")
            return None
        mioa_net = []
        influence_seed = [0 for i in range(self.node_num)]
        seed_sets = []
        for k in range(seed_num):
            # 假定选择第i号节点作为下一个选择的种子
            print("current times of circle：" + str(k))
            sigma_max = k + 1
            sigma_index = -1
            for candidate in range(len(candidates)):
                i = candidates[candidate]
                # 首先计算该节点的mioa与mioa_net的交集
                # 若该节点在之前已被选定为种子，就跳过此次循环
                if influence_seed[i] == 1:
                    continue
                else:
                    # 建立一个暂时的mioa_net为了暂存 mioa_net和当前遍历节点的mioa的交
                    current_mioa_net = list(mioa_net)
                    sigma = k + 1
                    ap = [0 for j in range(self.node_num)]
                    ap[i] = 1
                    for j in range(len(seed_sets)):
                        ap[seed_sets[j]] = 1
                    current_mioa_net = self.__merge_edge_set(current_mioa_net, self.mioa[i])

                    # 计算选i号节点作为此次添加的节点的时的提供的影响力
                    for nodes in range(len(current_mioa_net)):
                        end_node = current_mioa_net[nodes][1]
                        if (influence_seed[end_node] == 1) | (end_node == i):
                            continue
                        else:
                            ap[end_node] = self.__ap_calculate(end_node, ap, self.miia)
                            sigma = sigma + ap[end_node]

                    # 找到边界效益最大的点，记录下标
                    if sigma > sigma_max:
                        sigma_index = i
                        sigma_max = sigma

            if sigma_index != -1:
                seed_sets.append(sigma_index)
                influence_seed[sigma_index] = 1
                self.__merge_edge_set(mioa_net, self.mioa[sigma_index])
            else:
                print("can't find nodes which can create more influence")
                return seed_sets
        return seed_sets

    def influence(self, node_list):
        # 该点是否已被影响
        node_influenced = [0 for i in range(self.node_num)]
        # 影响点集合，第1个元素是点，第二个元素为是否进行过一次传播了
        curruent_influenced_node = []
        for i in range(len(node_list)):
            curruent_influenced_node.append([node_list[i], 0])
            node_influenced[node_list[i]] = 1

        step = 0
        while 1:
            # print (curruent_influenced_node)
            count_increase = 0
            length = len(curruent_influenced_node)
            for i in range(length):
                if curruent_influenced_node[i][1] == 0:
                    for j in range(len(self.__adjacent[curruent_influenced_node[i][0]][0])):
                        node = self.__adjacent[curruent_influenced_node[i][0]][0][j][0]
                        weight = self.__adjacent[curruent_influenced_node[i][0]][0][j][1]
                        if node_influenced[node] == 0:
                            value = random.random()
                            if value <= weight:
                                curruent_influenced_node.append([node, 0])
                                count_increase = count_increase + 1
                                node_influenced[node] = 1
                    curruent_influenced_node[i][1] = 1
                else:
                    continue
            step = step + 1
            if count_increase == 0:
                # print (curruent_influenced_node)
                return len(curruent_influenced_node)

    @staticmethod
    def __ap_calculate(u, ap_list, miia):
        product = 1
        for i in range(len(miia[u])):
            # 若两点间的最短路径仅含一条边，则两点之间一定是邻点
            if miia[u][i][3] == 1:
                product = product * (1 - ap_list[miia[u][i][0]] * miia[u][i][2])
        return 1 - product

    @staticmethod
    def __sorted_insert(list_t, tuple_t):
        length = tuple_t[3]
        flag_loc = 0
        for j in range(len(list_t)):
            if length < list_t[j][3]:
                flag_loc = 1
                list_t.insert(j, tuple_t)
                break
        if flag_loc == 0:
            list_t.append(tuple_t)

    def __merge_edge_set(self, mioa_net, mioa_i):
        # 合并两个边网络
        for i in range(len(mioa_i)):
            start_node = mioa_i[i][0]
            end_node = mioa_i[i][1]
            length = mioa_i[i][3]
            flag = 0
            for j in range(len(mioa_net)):
                if (start_node == mioa_net[j][0]) & (end_node == mioa_net[j][1]):
                    # 若该边已经在mioa_net中了，则更改其松弛次数为更多的那个，并重新插入
                    if length > mioa_net[j][3]:
                        mioa_net[j][3] = length
                        tmp = mioa_net[j]
                        mioa_net.remove(mioa_net[j])
                        self.__sorted_insert(mioa_net, tmp)
                    flag = 1
                    break

            # 倘若该边并没有出现在mioa_net中的话，将其按到seed的长度加入到mioa_net中
            if flag == 0:
                self.__sorted_insert(mioa_net, mioa_i[i])
        return mioa_net

    def get_miia_mioa(self, miia_path, mioa_path):
        self.miia = []
        self.mioa = []
        miia_file = open(miia_path, "r")
        mioa_file = open(mioa_path, "r")
        str = miia_file.readline()
        while (str != None) & (str != ""):
            tmp = str.split()
            miia = []
            for i in range(len(tmp)):
                item = tmp[i].split(",")
                miia.append([int(item[0]), int(item[1]), float(item[2]), int(item[3])])
            self.miia.append(miia)
            str = miia_file.readline()
        miia_file.close()

        str = mioa_file.readline()
        while (str != None) & (str != ""):
            tmp = str.split()
            mioa = []
            for i in range(len(tmp)):
                item = tmp[i].split(",")
                mioa.append([int(item[0]), int(item[1]), float(item[2]), int(item[3])])
            self.mioa.append(mioa)
            str = mioa_file.readline()
        mioa_file.close()

    def gen_learning_sets(self, file_out_path):
        file_out = open(file_out_path, "w")
        for i in range(len(self.__adjacent)):
            degree = len(self.__adjacent[i][0])
            weight_sum = 0
            for j in range(len(self.__adjacent[i][0])):
                weight_sum = weight_sum + self.__adjacent[i][0][j][1]
            file_out.write(str(degree) + " " + str(weight_sum) + "\n")
        file_out.close()


# 第一个参数是训练集文件，第二参数是测试集文件， 第三个参数是训练集的标签，返回的结果是测试集的标签预测结果
def svm_machine(file_train_path, file_test_path, class_table):
    file_in = open(file_test_path, "r")
    string = file_in.readline()
    test_sets = []
    while (string != None) & (string != ""):
        tmp = string.split()
        test_sets.append([float(tmp[0]), float(tmp[1])])
        string = file_in.readline()
    file_in = open(file_train_path, "r")
    string = file_in.readline()
    train = []
    while (string != None) & (string != ""):
        tmp = string.split()
        train.append([float(tmp[0]), float(tmp[1])])
        string = file_in.readline()
    clf = svm.SVC()
    clf.fit(train, class_table)
    arr = clf.predict(test_sets)
    result = []
    for j in range(len(arr)):
        if arr[j] == 1:
            result.append(j)
    return result


def MLP_classifier(file_train_path, file_test_path, class_table):
    file_in = open(file_test_path, "r")
    string = file_in.readline()
    test_sets = []
    while (string != None) & (string != ""):
        tmp = string.split()
        test_sets.append([float(tmp[0]), float(tmp[1])])
        string = file_in.readline()
    file_in = open(file_train_path, "r")
    string = file_in.readline()
    train = []
    while (string != None) & (string != ""):
        tmp = string.split()
        train.append([float(tmp[0]), float(tmp[1])])
        string = file_in.readline()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10, 10), random_state=1) # 三个隐藏层，每层5个节点
    clf.fit(train, class_table)
    arr = clf.predict(test_sets)
    result = []
    for j in range(len(arr)):
        if arr[j] == 1:
            result.append(j)
    return result


def linear_regression(file_train_path, file_test_path, class_table):
    file_in = open(file_test_path, "r")
    string = file_in.readline()
    test_sets = []
    while (string != None) & (string != ""):
        tmp = string.split()
        test_sets.append([float(tmp[0]), float(tmp[1])])
        string = file_in.readline()
    file_in = open(file_train_path, "r")
    string = file_in.readline()
    train = []
    while (string != None) & (string != ""):
        tmp = string.split()
        train.append([float(tmp[0]), float(tmp[1])])
        string = file_in.readline()
    clf = linear_model.LogisticRegressionCV()
    clf.fit(train, class_table)
    arr = clf.predict(test_sets)
    result = []
    for k in range(len(arr)):
        if arr[k] == 1:
            result.append(k)
    return result


def random_k_vertex(k, node_num):
    seeds = []
    count = 0
    while count < k:
        seed = int(random.random()*node_num)
        flag = 0
        for j in range(len(seeds)):
            if seeds[j] == seed:
                flag = 1
                break
        if flag == 0:
            seeds.append(seed)
            count = count + 1
    return seeds


net_test = Net("D:/train/bitcoin-OTC.txt",  "MIA", 6006)
# net_test = Net("D:/test/btc-alpha.txt", "MIA", 7605)
# net = Net("D:/WikiTalk.txt", "D:/result.txt", "MIA", 5, 10)
# net_test.gen_mia_model(0)

net_test.get_miia_mioa("D:/train/miia.txt", "D:/train/mioa.txt")

# start = time.clock()
# candidate = [i for i in range(net_test.node_num)]
# sets = net_test.mia_greedy(17, candidate)
# end = time .clock()
# print(sets)
# print(end - start)
# sum = 0
# for i in range(100):
#     sum = sum + net_test.influence(sets)
# print(sum)

for i in range(9):
    start = time.clock()
    candidate = [1, 7, 13, 35, 41, 202, 257, 304, 546, 905, 1018, 1317, 1334, 1352, 1386, 1396, 1565, 1566, 1615, 1810,
                 1899, 1953, 2028, 2045, 2067, 2125, 2266, 2296, 2388, 2635, 2642, 2942, 3129, 3451, 3649, 3722, 3735,
                 3828, 3897, 3988, 4172, 4197, 4291, 4559]
    sets = net_test.mia_greedy(i*2+1, candidate)
    end = time .clock()
    print(sets)
    print(end - start)
    sum = 0
    for j in range(100):
        sum = sum + net_test.influence(sets)
    print(sum)
    print()

# start = time.clock()
# randomsets = random_k_vertex(17, 6006)
# end = time.clock()
# print(randomsets)
# print(end - start)
# sum = 0
# for j in range(100):
#     sum = sum + net_test.influence(randomsets)
# print(sum)


# sets = [1, 8, 3, 13, 7, 4, 15, 177, 6, 5, 129, 2, 16, 10, 12, 11, 46, 9, 58, 33, 14, 38, 18, 30, 79,
#         22, 57, 798, 23, 61, 95, 25, 7564, 2336, 117, 43, 40, 69, 5342, 19, 26, 68, 17, 51, 7595, 178,
#         114, 45, 7603, 52, 145, 125, 28, 104, 239, 166, 42, 121, 36, 48, 130, 92, 102, 21, 50, 78, 37,
#         85, 65, 67, 89, 708, 128, 96, 86, 34, 49, 93, 55, 87, 115, 116, 206, 139, 32, 108, 109, 81, 84,
#         143, 39, 153, 252, 156, 64, 44, 71, 151, 491, 161]
# class_tables = [0 for i in range(7605)]
# for i in range(len(sets)):
#     class_tables[sets[i]] = 1
# result = linear_regression("D:/test/min_max_sets.txt", "D:/train/min_max_sets.txt", class_tables)
# print(result)
