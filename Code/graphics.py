import matplotlib.pyplot as plt


def make_heatmap(student_list, course_list, similarity_matrix, all_recommendations):
    for s_index, student in enumerate(student_list):
        student_rec_dict = dict(all_recommendations[student])
        for c_index, course in enumerate(course_list):
            similarity_matrix[s_index][c_index] = student_rec_dict.get(course, 0)

    plt.imshow(similarity_matrix)
    plt.xticks(range(len(course_list)), course_list, rotation=45, ha='right')
    plt.yticks(range(len(student_list)), student_list)
    plt.colorbar()
    plt.show()