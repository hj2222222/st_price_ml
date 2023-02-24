import pandas as pd
import numpy as np
#read_csv 안에 상대 경로를 넣어주세요. 만들 때 vs code에선 상대경로 사용시 오류가 나서 상대경로를 사용하지 않았습니다.
main_df=pd.read_csv('C:\programming_practice\python\python_programming\project\companydataexcludeUSIdx\Totaldata_000210.csv',index_col="Date")
sup_df=pd.read_csv('C:\programming_practice\python\python_programming\project\companydataexcludeUSIdx\supportindexscaled_000210.csv',index_col="Date")
merged_df=pd.merge(main_df,sup_df,on="Date")
merged_df=merged_df.drop([merged_df.columns[0],"ma_5","ma_20","ma_60","ma_120"],axis=1)