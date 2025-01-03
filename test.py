import numpy as np
import pandas as pd
# mapping_52_to_5 = {
#     0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2,
#     15: 2, 16: 2, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3,
#     28: 3, 29: 3, 30: 3, 31: 3, 32: 3, 33: 4, 34: 4, 35: 4, 36: 4, 37: 4, 38: 4, 39: 4, 40: 4,
#     41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4, 50: 4, 51: 4  # 最后一组映射到类别 5
# }
mapping_52_to_5 = {
    0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
    15: 3, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4, # 最后一组映射到类别 5
}
df_52 = pd.read_csv('confusion_matrix.csv', header=None)
# with open("cm.csv", "r") as cm:
#     df_52 = cm.read()
#     import ast
#     df_52 = ast.literal_eval(df_52)
#     df_52 = pd.DataFrame(df_52)
cm_5 = np.zeros((5, 5), dtype=int)
for i in range(24):
    for j in range(24):
        mapped_i = mapping_52_to_5[i]
        mapped_j = mapping_52_to_5[j]  
        cm_5[mapped_i, mapped_j] += df_52.iloc[i, j]
print("5分类混淆矩阵:")
print(cm_5)
cm_5 = np.array([[14009,2424,1625,915,390],
 [ 902,6974,1977,712,423],
 [ 406,2018,4363,1829,949],
 [ 128 ,628, 1236, 3446,  518],
 [  38,  200,  441,  632, 1475]])
accuracy = np.trace(cm_5) / np.sum(cm_5)
print(f"准确率 (Accuracy): {accuracy:.4f}")
precision = np.zeros(5)
recall = np.zeros(5)
f1_score = np.zeros(5)

for i in range(5):
    precision[i] = cm_5[i, i] / np.sum(cm_5[:, i]) if np.sum(cm_5[:, i]) != 0 else 0
    recall[i] = cm_5[i, i] / np.sum(cm_5[i, :]) if np.sum(cm_5[i, :]) != 0 else 0
    if precision[i] + recall[i] != 0:
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    else:
        f1_score[i] = 0

print("精确率 (Precision) per class:", precision)
print("召回率 (Recall) per class:", recall)
print("F1分数 (F1 Score) per class:", f1_score)

print(f"平均精确率 (Precision): {np.mean(precision):.4f}")
print(f"平均召回率 (Recall): {np.mean(recall):.4f}")
print(f"平均F1分数 (F1 Score): {np.mean(f1_score):.4f}")
