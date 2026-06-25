import string
import ast

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import pandas as pd
import numpy as np


def process_text(text):
    stemmer = PorterStemmer()

    trans = str.maketrans('','', string.punctuation)
    cleaned_text = text.translate(trans)

    tokens = word_tokenize(cleaned_text)
    filtered_words = [w for w in tokens if not w.lower() in stopwords.words('english')]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return stemmed_words



df_courses = pd.read_csv('../CSVs/Courses.csv')
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


df_students = pd.read_csv('../CSVs/Students.csv')
df_students['Marks'] = df_students['Marks'].apply(ast.literal_eval)

student_record = df_students.iloc[0]
student_marks = student_record['Marks']
print(student_marks)

student_vector = np.zeros(len(vocabulary))

for course_name, mark in student_marks.items():
    course_index = df_courses[df_courses['Course'] == course_name].index
    if not course_index.empty:
        idx = course_index[0]

        vector = vectors_courses[idx]

        student_vector += vector * (mark ** 2)

for i, course_vector in enumerate(vectors_courses):

    dot_product = np.dot(course_vector, student_vector)
    norm_course = np.linalg.norm(course_vector)
    norm_student = np.linalg.norm(student_vector)

    if norm_course == 0 or norm_student == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (norm_course * norm_student)

    course_name = df_courses.iloc[i]['Course']
    print(course_name + ' similarity: ' + str(similarity))
