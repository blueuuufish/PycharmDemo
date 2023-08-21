import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from model import GPT2, GPTNeo
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPTNeoForSequenceClassification, GPT2Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_model(args):
    print('Loading Model..')
    backbone_type = args.model_version
    device = 'cpu'
    weights_path = args.weights_path

    if backbone_type == 'gpt2-large':
        last_hidden_size = 1280
        backbone = GPT2Model.from_pretrained(backbone_type)
        model = GPT2(backbone, last_hidden_size=last_hidden_size)
    elif backbone_type == 'gpt2':
        last_hidden_size = 768
        backbone = GPT2Model.from_pretrained(backbone_type)
        model = GPT2(backbone, last_hidden_size=last_hidden_size)
    elif backbone_type == "EleutherAI/gpt-neo-125M":
        last_hidden_size = 768
        backbone = GPTNeoForSequenceClassification.from_pretrained(backbone_type)
        model = GPTNeo(backbone, last_hidden_size)

    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()
    print('Model Loaded.')

    return model, backbone_type, device


def metrics(pred, true):
    print("Prediction done...")


def evaluate_gpt(model, backbone_type, device, j, save_preds, input_sentence):
    print('Evaluating..')

    # 使用适当的tokenizer将输入句子转换为模型所需的格式
    tokenizer = GPT2Tokenizer.from_pretrained(backbone_type)
    encoded_input = tokenizer.encode_plus(input_sentence, return_tensors="pt", max_length=512, truncation=True)
    input_ids = encoded_input["input_ids"].to(device)

    with torch.no_grad():
        y_pred_test = F.sigmoid(model(input_ids).unsqueeze(0)).squeeze(0)
        print(f"Prediction result for '{input_sentence}': {y_pred_test.item()}")  # 打印预测结果

    metrics(None, None)  # 注意：这里没有真实的标签和预测，所以传入None

    if save_preds:
        save_type = "gpt-neo-125M" if backbone_type == 'EleutherAI/gpt-neo-125M' else backbone_type
        pd.DataFrame([y_pred_test.item()]).to_csv(f'./output/preds-{save_type}.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default='./output/gpt-neo-125M_3.pt', type=str,
                        help="Trained weights to evaluate.")
    parser.add_argument('-j', default=0.5708000063896179, type=float, help='Threshold for optimisation.')
    parser.add_argument("--model_version", default='EleutherAI/gpt-neo-125M', type=str,
                        help="Backbone version of GPT models available.")
    parser.add_argument("--save_preds", default=True, help="Whether to save predictions")
    parser.add_argument('--input_sentence', required=True, type=str, help="Sentence to evaluate.")
    args = parser.parse_args()

    model, backbone_type, device = load_model(args)
    evaluate_gpt(model, backbone_type, device, args.j, args.save_preds, args.input_sentence)


if __name__ == '__main__':
    main()

# ------------------------------------------------line from here-----------------------------------------------------
#
# # script arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--weights_path', default='./output/gpt-neo-125M_3.pt', required=False, type=str, help="Trained weights to evaluate.")
# parser.add_argument('-j', default=0.5708000063896179, required=False, type=float, help='Thresold for optimisation.')
# parser.add_argument("--model_version", default='EleutherAI/gpt-neo-125M', required=False, type=str, help="Backbone version of GPT models available.")
# parser.add_argument("--save_preds", default=True, required=False, help="Whether to save predictions")
# args = parser.parse_args()
#
# print('Loading Model..')
# backbone_type = args.model_version
# device = 'cpu'
# j=args.j
# weights_path = args.weights_path
# dataset = FakeReviewsDataset("./data/test.csv", backbone_type)
# test_dataloader = DataLoader(dataset, 1, shuffle=False)
#
# if backbone_type == 'gpt2-large':
#     last_hidden_size = 1280
#     backbone = GPT2Model.from_pretrained(backbone_type)
#     model = GPT2(backbone, last_hidden_size=last_hidden_size)
# if backbone_type == 'gpt2':
#     last_hidden_size = 768
#     backbone = GPT2Model.from_pretrained(backbone_type)
#     model = GPT2(backbone, last_hidden_size=last_hidden_size)
# if backbone_type == "EleutherAI/gpt-neo-125M":
#     last_hidden_size = 768
#     backbone = GPTNeoForSequenceClassification.from_pretrained(backbone_type)
#     model = GPTNeo(backbone, last_hidden_size)
#
# model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
# model.eval()
# print('Model Loaded.')
#
# def metrics(pred, true):
#     print("hello prediction...")
#
# def evaluate_gpt(j = j, save_preds = args.save_preds):
#     print('Evaluating..')
#     with torch.no_grad():
#         predictions = []
#         true = []
#         for data_test in test_dataloader:
#             X_test = data_test['input']['input_ids'].to(device)
#             y_test = data_test['label'].numpy()
#             y_pred_test = F.sigmoid(model(X_test).unsqueeze(0)).squeeze(0)#.detach().cpu().numpy()
#             predictions.append(y_pred_test)
#             true.append(y_test)
#     metrics(predictions, true)
#     if backbone_type == 'EleutherAI/gpt-neo-125M':
#         save_type = "gpt-neo-125M"
#     else:
#         save_type = backbone_type
#     if save_preds:
#         pd.DataFrame(predictions).to_csv(f'./output/preds-{save_type}.csv', index = False)
#
#
#
# def main():
#     evaluate_gpt()
#
# if __name__ == '__main__':
#     main()

# ------------------------------------------------line from here-----------------------------------------------------

# from model import GPT2, GPTNeo
# from dataset import FakeReviewsDataset
# from torch.utils.data import DataLoader
# from transformers import GPT2Model, GPTNeoForSequenceClassification
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import torch 
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import argparse

# # script arguments 
# parser = argparse.ArgumentParser()
# parser.add_argument('--weights_path', default='./output/gpt-neo-125M_2.pt', required=False, type=str, help="Trained weights to evaluate.")
# parser.add_argument('-j', default=0.5708000063896179, required=False, type=float, help='Thresold for optimisation.')
# parser.add_argument("--model_version", default='EleutherAI/gpt-neo-125M', required=False, type=str, help="Backbone version of GPT models available.")
# parser.add_argument("--save_preds", default=False, required=False, help="Whether to save predictions")
# args = parser.parse_args()

# print('Loading Model..')
# backbone_type = args.model_version
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# j=args.j
# weights_path = args.weights_path
# dataset = FakeReviewsDataset("./data/test.csv", backbone_type)
# test_dataloader = DataLoader(dataset, 1, shuffle=False)

# if backbone_type == 'gpt2-large':
#     last_hidden_size = 1280
#     backbone = GPT2Model.from_pretrained(backbone_type)
#     model = GPT2(backbone, last_hidden_size=last_hidden_size)
# if backbone_type == 'gpt2':
#     last_hidden_size = 768
#     backbone = GPT2Model.from_pretrained(backbone_type)
#     model = GPT2(backbone, last_hidden_size=last_hidden_size)
# if backbone_type == "EleutherAI/gpt-neo-125M":
#     last_hidden_size = 768
#     backbone = GPTNeoForSequenceClassification.from_pretrained(backbone_type)
#     model = GPTNeo(backbone, last_hidden_size)

# model.load_state_dict(torch.load(weights_path, map_location=torch.device(device))) 
# model.eval()
# print('Model Loaded.')

# def metrics(pred, true):
#     acc = round(accuracy_score(true, pred) * 100, 2)
#     prec = round(precision_score(true, pred) * 100, 2)
#     rec = round(recall_score(true, pred) * 100, 2)
#     f1 = round(f1_score(true, pred) * 100, 2)
#     print('Evaluation metrics on test set: ')
#     print('Accuracy: ', acc, "%")
#     print('Precision: ', prec, "%")
#     print('Recall: ', rec, "%")
#     print('F1-score', f1, "%")

# def evaluate_gpt(j = j, save_preds = args.save_preds):
#     print('Evaluating..')
#     with torch.no_grad():
#         predictions = []
#         true = []
#         i = 0
#         for data_test in test_dataloader:
#             print(i)
#             i = i + 1
#             X_test = data_test['input']['input_ids'].to(device)
#             y_test = data_test['label'].numpy()
#             y_pred_test = F.sigmoid(model(X_test).unsqueeze(0)).squeeze(0)#.detach().cpu().numpy()
#             y_pred_test = np.where(y_pred_test > j, 1, 0)
#             predictions.append(y_pred_test)
#             true.append(y_test)    
#     metrics(predictions, true)
#     if backbone_type == 'EleutherAI/gpt-neo-125M':
#         save_type = "gpt-neo-125M"
#     else:
#         save_type = backbone_type
#     if save_preds:
#         pd.DataFrame(predictions).to_csv(f'./output/preds-{save_type}.csv', index = False)


# def main():
#     evaluate_gpt()

# if __name__ == '__main__':
#     main()
