from transformers import AutoTokenizer

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def load_tokenizer(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path)