import pandas as pd

# Chargement du fichier CSV
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Affichage des 5 premières lignes
print(df.head())
