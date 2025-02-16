import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

file="C:/Users/ashwi/Desktop/Sem 4/ML/Lab_ML/Lab Session Data.xlsx"
data=pd.read_excel(file, sheet_name="thyroid0387_UCI")

binary=["on thyroxine", "query on thyroxine", "on antithyroid medication",
               "sick", "pregnant", "thyroid surgery", "I131 treatment", "query hypothyroid",
               "query hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary",
               "psych", "TSH measured", "T3 measured", "TT4 measured", "T4U measured",
               "FTI measured", "TBG measured"]
data[binary]=data[binary].replace({'f': 0, 't': 1}).astype(int)

vec1=data.iloc[0][binary]
vec2=data.iloc[1][binary]

f11=sum((vec1 == 1) & (vec2 == 1))  # Both 1
f00=sum((vec1 == 0) & (vec2 == 0))  # Both 0
f10=sum((vec1 == 1) & (vec2 == 0))  # vec1 = 1, vec2 = 0
f01=sum((vec1 == 0) & (vec2 == 1))  # vec1 = 0, vec2 = 1

JC=f11/(f01+f10+f11)
SMC=(f11+ f00)/(f00+f01+f10+f11)

print(f"Jaccard Coefficient: {JC:.4f}")
print(f"Simple Matching Coefficient: {SMC:.4f}")


#A9
vec1 = vec1.values.reshape(1, -1)
vec2 = vec2.values.reshape(1, -1)

cos=cosine_similarity(vec1, vec2)[0][0]
print(f"Cosine Similarity: {cos:.4f}")

