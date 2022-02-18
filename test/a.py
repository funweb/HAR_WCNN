import pandas as pd
import numpy as np
df = pd.DataFrame(np.arange(20).reshape(5,4),index=list("ABCDE"),columns=list("WXYZ"))

print(df)
print(df.loc[:, "Y"])

index_list = ["0", "1", "2", "acc_mean", "acc_std"]

df_2 = pd.DataFrame(index=index_list)

df_2["999"] = df.loc[:, "Y"].tolist()

print(df_2)

