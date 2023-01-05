import numpy as np

def invocation_result_to_readable_list(examples, result, label_masks):
    labels = ['o', "ingredient_b", "ingredient-i", "ingredient-o", "tool", "time_b", "time_i", "time_o"]

    tag_to_index = {tag: index for index, tag in enumerate(labels)}
    index_to_tag = {index: tag for index, tag in enumerate(labels)}

    predictions = []

    # 추론 결과인 각 토큰 별 label에 해당할 확률값을 해당하는 라벨 index로 해석하기
    # (미역, (0.1, 0.8, 0.2, 0.2, 0, 0, 0, 0)) -> (미역, 1)
    for i in result["output_1"].numpy():
        temp = []
        for j in i:
            temp.append(np.argmax(j))

        predictions.append(temp)

    pred_list = []
    result_list = []

    # 각 토큰 별로 확률이 가장 큰 라벨 index를 실제 라벨로 해석하기
    # (미역, 1) -> (미역, ingredient_b)
    for i in range(0, len(label_masks)):
        pred_tag = []
        for label_index, pred_index in zip(label_masks[i], predictions[i]):
            if label_index != -100:
                pred_tag.append(index_to_tag[pred_index])

        pred_list.append(pred_tag)

    for example, pred in zip(examples, pred_list):
        one_sample_result = []
        for one_word, label_token in zip(example, pred):
            one_sample_result.append((one_word, label_token))
        result_list.append(one_sample_result)

    return result_list

# (단어, 라벨) 형태로 해석되어 있는 결과값을 {식재료 : [], 도구 : [], 시간 : []} 형태로 변경
def result_list_to_target_word_list(result_list):
    ingredient = []
    tool = []
    time = []

    for index, k in enumerate(result_list):
        temp_ingredient = ''
        temp_time = ''
        in_ingredient = False
        in_time = False

        for i in range(len(k)):
            # ingredient_b는 ingredient-i나 ingredient-o가 이어진다면 한 단어로 묶어 식재료 리스트에 추가
            if k[i][1] == "ingredient_b":
                in_ingredient = True
                temp_ingredient = k[i][0]
            elif in_ingredient and k[i][1] in ["ingredient-i", "ingredient-o"]:
                temp_ingredient += ' ' + k[i][0]
            elif in_ingredient and k[i][1] not in ["ingredient-i", "ingredient-o"]:
                ingredient.append(temp_ingredient)
                in_ingredient = False
            # tool은 BIO 태그가 아닌 단독으로 존재하므로 등장하면 도구 리스트에 추가
            elif k[i][1] == "tool":
                tool.append(k[i][0])
            # time_b는 time_i나 time_o가 이어진다면 한 단어로 묶어 시간 리스트에 추가
            elif k[i][1] == "time_b":
                in_time = True
                temp_time = k[i][0]
            elif in_time and k[i][1] in ["time_i", "time_o"]:
                temp_time += ' ' + k[i][0]
            elif in_time and k[i][1] not in ["time_i", "time_o"]:
                time.append((index + 1, temp_time))
                in_time = False

    result = {
        "ingredients": list(set(ingredient)),
        "tools": list(set(tool)),
        "times": time
    }

    return result