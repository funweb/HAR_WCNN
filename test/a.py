import pandas as pd
import itertools  # 排列组合
from CVTest5x2 import five_two_statistic

if __name__ == '__main__':
    data_df = pd.read_csv("tmp.csv")

    random_seeds = [26, 43, 7, 13,  35,]

    distants = [9999, 999, 1, 2, 3, 4, 5]

    method_condition = "WCNN"
    datasets = ["cairo", "milan", "kyoto7", "kyoto8", "kyoto11"]

    for dataset_condition in datasets:
        print("\n\n\n")
        # 通过 方法 和 数据集 约束数据
        condition_df = data_df[data_df['method'].str.contains(method_condition)]
        condition_df = condition_df[condition_df['dataset'].str.contains(dataset_condition)]

        # 通过 0/1 交叉约束数据
        condition_df_0 = condition_df[condition_df["k"] == 0]
        condition_df_1 = condition_df[condition_df["k"] == 1]

        # 分别计算
        for ds_A, ds_B in itertools.combinations(distants, 2):  # 各种组合
            p_1_list = []
            p_2_list = []
            for rs in random_seeds:  # 也就是5次交叉结果
                p_A_0 = condition_df_0[condition_df_0["random"] == rs][str(ds_A)].values[0]
                p_B_0 = condition_df_0[condition_df_0["random"] == rs][str(ds_B)].values[0]
                p_1 = p_A_0 - p_B_0
                p_1_list.append(p_1)

                p_A_1 = condition_df_1[condition_df_1["random"] == rs][str(ds_A)].values[0]
                p_B_1 = condition_df_1[condition_df_1["random"] == rs][str(ds_B)].values[0]
                p_2 = p_A_1 - p_B_1
                p_2_list.append(p_2)

            # print("5x2 CV Paired t-test")
            t, p = five_two_statistic(p_1_list, p_2_list)
            # print(f"pair: {ds_A, ds_B}, t statistic: {t}, p-value: {p}\n")
            if ds_A == 9999 and ds_B != 999:
                if t < 0:
                    print(f"dataset: {dataset_condition}, pair: {ds_A, ds_B}, t statistic: {t}, p-value: {p}")
