import ast
import pandas as pd
import recommender


df_courses = pd.read_csv('../CSVs/Courses.csv')
df_students = pd.read_csv('../CSVs/Students.csv')
df_students['Marks'] = df_students['Marks'].apply(ast.literal_eval)

vectors_courses, vocabulary = recommender.build_course_matrix(df_courses)

for index, row in df_students.iterrows():
    recommender.calculate_recommendations(row['Student'], row['Marks'], df_courses, vectors_courses, vocabulary)

