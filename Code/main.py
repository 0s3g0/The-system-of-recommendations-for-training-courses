import ast

import numpy as np
import pandas as pd
import recommender
import graphics

df_courses = pd.read_csv('../CSVs/Courses.csv')
df_students = pd.read_csv('../CSVs/Students.csv')
df_students['Marks'] = df_students['Marks'].apply(ast.literal_eval)

vectors_courses, vocabulary = recommender.build_course_matrix(df_courses)
all_recommendations = {}

for index, row in df_students.iterrows():
    student_recommendations = recommender.calculate_recommendations(row['Marks'], df_courses, vectors_courses, vocabulary)
    all_recommendations[row['Student']] = student_recommendations


for student_name, recommendations_list in all_recommendations.items():
    print("-" * 50)
    print(f"Recommendations for student: {student_name} \n")

    for course_name, similarity in recommendations_list[:5]:
        print(f"{course_name} similarity: {similarity:.3f}")

student_list = list(all_recommendations.keys())
course_list = list(df_courses['Course'].unique())

similarity_matrix = np.zeros((len(student_list), len(course_list)))

graphics.make_heatmap(student_list, course_list, similarity_matrix, all_recommendations)