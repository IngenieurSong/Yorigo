import json
from transformers import BertTokenizer
import tensorflow as tf
from preprocessing import convert_examples_to_features_for_prediction, remove_stopword
from postprocessing import invocation_result_to_readable_list, result_list_to_target_word_list

def serverless_pipeline(tokenizer_path = "./tokenizer/", model_path = "./model/informationFromRecipe_20000"):
    # Initializes the model and tokenzier and returns a predict function that ca be used as pipeline
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only = True)
    model = tf.saved_model.load(model_path)
    inference_func = model.signatures["serving_default"]

    def predict(recipe_explanation):
        # remove stopword such as postposition, meanless word from recipe data
        sentences = remove_stopword(recipe_explanation)
        sentences = [sent.split() for sent in sentences]
        # Add [CLS], [SEP], [POS] tokens to sentences to match NER Task Input in BERT
        X_pred, label_mask = convert_examples_to_features_for_prediction(sentences, 32, tokenizer)
        # invoke
        result = inference_func(input_1 = X_pred[0], input_2 = X_pred[1], input_3 = X_pred[2])
        # Convert invocation result to Human-understandable result
        result_list = invocation_result_to_readable_list(sentences, result, label_mask)
        result = result_list_to_target_word_list(result_list)

        return result

    return predict

# initializes the pipeline
named_entity_recognition_pipeline = serverless_pipeline()

# Receive raw data from client and send it to NER model pipeline
def handler(event, context):
    try:
        # loads the incoming event into a dictonary
        # uses the pipeline to predict the answer
        body = json.loads(event["body"])
        result = named_entity_recognition_pipeline(recipe_explanation = body["recipe_explanations"])
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"result" : result}, ensure_ascii = False).encode("utf-8")
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }