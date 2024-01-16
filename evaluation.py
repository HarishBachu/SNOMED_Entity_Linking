import numpy as np 
import pandas as pd 
import argparse

def micro_iou(df_true, df_pred, note_id):
  # Slice ground truth and predictions for specific class (note_id) 
  df_true = df_true[df_true["note_id"]==note_id]
  df_pred = df_pred[df_pred["note_id"]==note_id]

  # Find distinct list of extracted entities 
  y_true = df_true["concept_id"].tolist() 
  y_pred = df_pred["concept_id"].tolist() 

  return np.intersect1d(y_true, y_pred).shape[0] / np.union1d(y_true, y_pred).shape[0] 

def macro_iou(df_true, df_pred):
  #Initialize empty array for micro IOU for each class 
  iou = []

  #Calculate Macro IOU as mean of micro IOUs
  for note_id in df_true["note_id"].unique():
    iou.append(micro_iou(df_true, df_pred, note_id))
  return np.mean(iou)

DEFAULT_FILE = "test.csv"

arg_parser = argparse.ArgumentParser(prog='Model Evaluation', description='Evaluates NER models based on Macro IOU score')

arg_parser.add_argument('-g', '--ground_truth', default=DEFAULT_FILE, help='Ground Truth data as csv')
arg_parser.add_argument('-p', '--predicted', default=DEFAULT_FILE, help="Model Predictions as csv")

custom_mapping = pd.read_csv("custom_mapping.csv")

_args = arg_parser.parse_args()
df_true = pd.read_csv(_args.ground_truth)
df_pred = pd.read_csv(_args.predicted)

df_true = pd.merge(df_true, custom_mapping, on="note_id", how="left")
df_true["note_id"] = df_true["re_mapped"]
df_true = df_true.drop(columns=["re_mapped"])

df_true = df_true[df_true["note_id"].isin(df_pred["note_id"].unique())]

print(macro_iou(df_true, df_pred))