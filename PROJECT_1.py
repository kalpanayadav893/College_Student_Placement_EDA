    # Project Title: Exploratory Data Analysis (EDA) on a Public Set
# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

# 2. Load Dataset
df = pd.read_csv(r"C:\Users\kalpa\Downloads\college_student_placement_dataset.csv")

# 3. Initial Data Overview
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())
print("\nSummary Statistics:\n", df.describe(include='all'))
print("\nUnique Values in Each Column:\n", df.nunique())

# unique values
if 'College_ID' in df.columns:
    print("\nCollege_ID unique values:\n", df['College_ID'].unique())

if 'CGPA' in df.columns:
    print("\nCGPA unique values:\n", df['CGPA'].unique())

if 'Academic_Performance' in df.columns:
    print("\nAcademic_Performance unique values:\n", df['Academic_Performance'].unique())

# 4. Missing Values Analysis
print("\nMissing Values per Column:\n", df.isnull().sum())

# Visualizing Missing Values
msno.matrix(df)
plt.title("Missing Values Matrix")
plt.show()

msno.heatmap(df)
plt.title("Missing Values Heatmap")
plt.show()

# 5. Drop Irrelevant or Redundant Columns
columns_to_drop = [col for col in ['College_ID', 'CGPA'] if col in df.columns]
student = df.drop(columns=columns_to_drop)
print("\nDataset After Dropping Columns:\n", student.head())

# 6. Correlation Analysis
numeric_cols = student.select_dtypes(include=[np.number])
correlation = numeric_cols.corr()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# 7. Pairplot
if 'Placed' in student.columns:
    sns.pairplot(student, hue='Placed')
    plt.suptitle("Pairplot with Placement Status", y=1.02)
    plt.show()
else:
    sns.pairplot(student.select_dtypes(include=[np.number]))
    plt.suptitle("Pairplot of Numeric Features", y=1.02)
    plt.show()

# 8. Outlier Detection (Boxplots)
for col in numeric_cols.columns:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.tight_layout()
    plt.show()

# 9. Distribution Plots
# Internship Experience
if 'Internship_Experience' in student.columns:
    sns.displot(student['Internship_Experience'], kde=True)
    plt.title("Distribution of Internship Experience")
    plt.xlabel("Internship Experience")
    plt.ylabel("Count")
    plt.show()

# CGPA from original df
if 'CGPA' in df.columns:
    sns.displot(df['CGPA'], bins=10, kde=True)
    plt.title("Distribution of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Count")
    plt.show()

# 10. Categorical Value Counts and Countplots
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"\nValue Counts for {col}:\n", df[col].value_counts())
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 11. Boxplots of Key Categorical Relationships
if {'CGPA', 'Academic_Performance'}.issubset(df.columns):
    sns.catplot(x='Academic_Performance', y='CGPA', kind='box', data=df)
    plt.title("Boxplot of CGPA by Academic Performance")
    plt.xlabel("Academic Performance")
    plt.ylabel("CGPA")
    plt.xticks(rotation=45)
    plt.show()

if {'Internship_Experience', 'Academic_Performance'}.issubset(df.columns):
    sns.catplot(x='Academic_Performance', y='Internship_Experience', kind='box', data=df)
    plt.title("Boxplot of Internship Experience by Academic Performance")
    plt.xlabel("Academic Performance")
    plt.ylabel("Internship Experience")
    plt.xticks(rotation=45)
    plt.show()

# 12. Plot using Plotly
if {'CGPA', 'IQ', 'Academic_Performance'}.issubset(df.columns):
    fig = px.scatter(df, x='CGPA', y='IQ', color='Academic_Performance', 
                    title='Scatter: CGPA vs IQ by Academic Performance')
    fig.show()

# 13. Statistical Summary
print("\nSkewness of Numeric Features:\n", df[numeric_cols.columns].skew())
print("\nKurtosis of Numeric Features:\n", df[numeric_cols.columns].kurt())

# 14. Final Summary - Key Insights 
"""
Final Insights:
1. CGPA is positively associated with Internship Experience and IQ.
2. Placement status shows strong correlation with Academic Performance and CGPA.
3. Several features exhibit mild skewness, suggesting some transformation may help.
4. Outliers detected in Internship Experience and IQ—could impact model training.
5. Students with better Academic Performance tend to have higher CGPA and placement rates.
"""

