import numpy as np

def convert_examples_to_features_for_prediction(recipe, max_seq_len, tokenizer,
                                 pad_token_id_for_segment = 0, pad_token_id_for_label = -100):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, label_masks = [], [], [], []

    for example in recipe:
        tokens = []
        label_mask = []
        for one_word in example:
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            label_mask.extend([0]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

        # discard if the number of tokens exceeds the maximum input sequence length
        special_tokens_count = 2
        if(len(tokens) > max_seq_len - special_tokens_count):
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            label_mask = label_mask[:(max_seq_len - special_tokens_count)]

        # adding [SEP] token to input token sequence
        tokens += [sep_token]
        label_mask += [pad_token_id_for_label]

        # adding [CLS] token to input token sequence
        tokens = [cls_token] + tokens
        label_mask = [pad_token_id_for_label] + label_mask

        # give token id and give attention mask 1 to indicate that it is not padding
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        # adding padding to match the length of different input sequences and giving an attention mask to indicate that it is a meaningless sentence
        padding_count = max_seq_len - len(input_id)
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        # give token type id to distinguish whether input sequence is two or one sentence
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label_mask = label_mask + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
        assert len(label_mask) == max_seq_len, "Error with labels length {} vs {}".format(len(label_mask), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        label_masks.append(label_mask)

    input_ids = np.array(input_ids, dtype = np.int32)
    attention_masks = np.array(attention_masks, dtype = np.int32)
    token_type_ids = np.array(token_type_ids, dtype = np.int32)
    label_masks = np.asarray(label_masks, dtype = np.int32)

    return (input_ids, attention_masks, token_type_ids), label_masks

def remove_stopword(recipe_explanations):
    # define stopword to remove
    stopword = ['을', '를', '와', '과', '.', ',', '에', "으로", '는', "정도", "반죽"]
    sentence = []

    for sent in recipe_explanations:
        for word in sent:
            if(word in stopword):
                # remove stopword
                sent = sent.replace(word, ' ')

        sentence.append(sent)

    return sentence