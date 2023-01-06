from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_scene_steps(recipe_steps, sentences, recipe_name):
    i = 0
    j = 0

    step_result = []
    tfidf_vectorizer = TfidfVectorizer()

    # recipe_steps 리스트에 요리 준비과정 구별을 위한 none, recipe_name 문자열 추가
    recipe_steps.insert(0, "none " + recipe_name)
    recipe_steps = recipe_steps + ["padding", "padding"]

    # recognition된 자막과 recipe_steps 내용 비교를 통해서 단계 추론
    while(i < len(recipe_steps) - 1 and j != len(sentences) - 1):
        target_sentence = sentences[j]

        # recognition된 자막과 recipe_steps의 현재 단계 설명에 대한 cosine similarity 비교
        # recognition된 자막과 recipe_steps의 다음 단계 설명에 대한 cosine similarity 비교
        tfidf_matrix_1 = tfidf_vectorizer.fit_transform((target_sentence, recipe_steps[i]))
        cos_similar_1 = cosine_similarity(tfidf_matrix_1[0:1], tfidf_matrix_1[1:2])
        tfidf_matrix_2 = tfidf_vectorizer.fit_transform((target_sentence, recipe_steps[i + 1]))
        cos_similar_2 = cosine_similarity(tfidf_matrix_2[0:1], tfidf_matrix_2[1:2])
        tfidf_matrix_3 = tfidf_vectorizer.fit_transform((target_sentence, recipe_steps[i + 2]))
        cos_similar_3 = cosine_similarity(tfidf_matrix_3[0:1], tfidf_matrix_3[1:2])

        max_similarity = max(cos_similar_1, cos_similar_2, cos_similar_3)

        # 각 scene의 프레임 별로 해당할 확률이 가장 높은 레시피 단계 배정
        if((cos_similar_1 == cos_similar_2 and cos_similar_2 == cos_similar_3) or (max_similarity == cos_similar_1)):
            step_result.append(i)
        elif(max_similarity == cos_similar_2):
            step_result.append(i + 1)
            i += 1
        elif(max_similarity == cos_similar_3):
            step_result.append(i + 2)
            i += 2

        j += 1

        if(i == len(recipe_steps) - 1):
            for _ in range(len(sentences) - 1 - j):
                # print(i)
                step_result.append(i)
            break

    return step_result

# 같은 단계에 있는 scene들의 timestemp 합치기
# ex)start time과 end time이 [00:00:00, 00:00:07]인 scene 1과 [00:00:07, 00:00:13]인 scene 2가 같은 단계로 추론된다면
# 두 scene의 timestemp를 합쳐서 [00:00:00, 00:00:13]으로 만들기
def combine_scene_timestemp(step_result, scene_time_list):
    combine_result = []
    start, end = 0, 1

    while(end < len(step_result)):
        if(step_result[start] == step_result[end]):
            end += 1
        else:
            combine_result.append((step_result[start], scene_time_list[start][0], scene_time_list[end - 1][1]))
            start = end
            end += 1
    else:
        combine_result.append((step_result[start], scene_time_list[start][0], scene_time_list[end - 1][1]))

    return combine_result