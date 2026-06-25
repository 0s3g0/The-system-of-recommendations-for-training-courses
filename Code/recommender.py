import numpy as np

from text_processing import process_text


def build_course_matrix(df_courses):
    df_courses['Full text'] = df_courses['Course'] + ' ' + df_courses['Description']
    courses_text = list(df_courses['Full text'])

    filtered_courses = []
    for course in courses_text:
        filtered_courses.append(process_text(course))

    all_words = []
    for word in filtered_courses:
        all_words.extend(word)

    vocabulary = sorted(list(set(all_words)))
    vectors_courses = np.zeros((len(filtered_courses), len(vocabulary)))

    for i, course_words in enumerate(filtered_courses):
        for j, word in enumerate(vocabulary):
            vectors_courses[i, j] = course_words.count(word)

    return vectors_courses, vocabulary


def calculate_recommendations(student_marks, df_courses, vectors_courses, vocabulary):
    student_vector = np.zeros(len(vocabulary))

    for course_name, mark in student_marks.items():
        course_index = df_courses[df_courses['Course'] == course_name].index
        if not course_index.empty:
            idx = course_index[0]
            vector = vectors_courses[idx]
            student_vector += vector * (mark ** 2)

    recommendations = []
    for i, course_vector in enumerate(vectors_courses):
        course_name = df_courses.iloc[i]['Course']

        if course_name in student_marks:
            continue

        dot_product = np.dot(course_vector, student_vector)
        norm_course = np.linalg.norm(course_vector)
        norm_student = np.linalg.norm(student_vector)

        if norm_course == 0 or norm_student == 0: similarity = 0.0
        else: similarity = dot_product / (norm_course * norm_student)

        course_name = df_courses.iloc[i]['Course']
        recommendations.append([course_name, similarity])

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations