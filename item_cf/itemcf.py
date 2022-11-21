import math


class ItemBasedCF:
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = []
        self.train_data = {}
        self.item_similarity_matrix = []

    def read_data(self):
        """
            读取元数据
        """
        datas = []
        for line in open(self.datafile):
            userid, itemid, count = line.rstrip("\n").split(",")
            datas.append((userid, itemid, count))
        self.data = datas

    def pre_process_data(self):
        """
            预处理数据
        """
        train_data = {}
        for user, item, count in self.data:
            train_data.setdefault(user, {})
            train_data[user][item] = count
        self.train_data = train_data

    def item_similarity(self):
        """
            生成用户相似度矩阵
        """
        self.item_similarity_matrix = dict()
        item_item_matrix = dict()
        item_user_matrix = dict()

        for user, items in self.train_data.items():
            for item_id, score in items.items():
                item_user_matrix.setdefault(item_id, 0)
                item_user_matrix[item_id] += 1
                item_item_matrix.setdefault(item_id, {})
                for i in items.keys():
                    if i == item_id:
                        continue
                    item_item_matrix[item_id].setdefault(i, 0)
                    item_item_matrix[item_id][i] += 1 / math.log(1 + len(items) * 1.0)

        for item_id, related_items in item_item_matrix.items():
            self.item_similarity_matrix.setdefault(item_id, dict())
            for related_item_id, count in related_items.items():
                self.item_similarity_matrix[item_id][related_item_id] = count / math.sqrt(
                    item_user_matrix[item_id] * item_user_matrix[related_item_id])
            if len(self.item_similarity_matrix[item_id].values()) == 0:
                continue
            sim_max = max(self.item_similarity_matrix[item_id].values())
            for item in self.item_similarity_matrix[item_id].keys():
                self.item_similarity_matrix[item_id][item] /= sim_max

    def recommend(self, user_id, k, n):
        """
            给用户推荐物品列表
        """
        rank = dict()
        interacted_items = self.train_data.get(user_id, {})
        for item_id, score in interacted_items.items():
            for i, sim_ij in sorted(self.item_similarity_matrix[item_id].items(), key=lambda x: float(x[1]),
                                    reverse=True)[0:k]:
                if i in interacted_items.keys():
                    continue
                rank.setdefault(i, 0)
                rank[i] += float(score) * sim_ij
        return dict(sorted(rank.items(), key=lambda x: float(x[1]), reverse=True)[0:n])


if __name__ == "__main__":
    item_based_cf = ItemBasedCF("/Users/gang.song/Desktop/result2.csv")
    item_based_cf.read_data()
    item_based_cf.pre_process_data()
    item_based_cf.item_similarity()

    while True:
        user_id = input("please enter the userID: ")
        k = input("please enter K: ")
        n = input("please enter N: ")
        print("-- recommendations --")
        result = item_based_cf.recommend(user_id, int(k), int(n))
        for k, v in result.items():
            print(f"{k} {v}")
