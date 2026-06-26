import matplotlib.pyplot as plt
import numpy as np


# Создание тепловой карты
def make_heatmap(student_list, course_list, all_recommendations):
    # Пустая матрица рекомендаций
    similarity_matrix = np.zeros((len(student_list), len(course_list)))

    for s_index, student in enumerate(student_list): # Проходимся по студентам
        student_rec_dict = dict(all_recommendations[student])
        for c_index, course in enumerate(course_list):
            similarity_matrix[s_index][c_index] = student_rec_dict.get(course, 0) # Заполняем матрицу значениями

    plt.imshow(similarity_matrix) # Строим карту
    plt.xticks(range(len(course_list)), course_list, rotation=45, ha='right') # Добавляем значения на оси
    plt.yticks(range(len(student_list)), student_list)
    plt.colorbar()
    plt.show()