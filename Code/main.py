import ast

import pandas as pd

import recommender
import graphics

# Загружаем наши таблицы
df_courses = pd.read_csv('../CSVs/Courses.csv')
df_students = pd.read_csv('../CSVs/Students.csv')
df_students['Marks'] = df_students['Marks'].apply(ast.literal_eval) # Превращаем оценки в массив

vectors_courses, vocabulary = recommender.build_course_matrix(df_courses) # Создаем матрицу из векторов курсов
all_recommendations = {}

# Вычисляем рекомендации для определенного студента и добавляем в словарь вида {ФИО: [список рекомендаций]}
for index, row in df_students.iterrows():
    student_recommendations = recommender.calculate_recommendations(row['Marks'], df_courses, vectors_courses, vocabulary)
    all_recommendations[row['Student']] = student_recommendations

# Вывод студентов в консоль
for student_name, recommendations_list in all_recommendations.items():
    print("-" * 50)
    print(f"Recommendations for student: {student_name} \n")

    for course_name, similarity in recommendations_list[:5]:
        print(f"{course_name} similarity: {similarity:.4f}")

# Подготовка к созданию тепловой карты
# Листы с именами\названиями для осей графика
student_list = list(all_recommendations.keys())
course_list = list(df_courses['Course'].unique())

graphics.make_heatmap(student_list, course_list, all_recommendations)