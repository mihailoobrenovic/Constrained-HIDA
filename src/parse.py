# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="space", help="For now, space or rgbd")
parser.add_argument("--num_runs",type=int, default=1, help="num_runs")
parser.add_argument("--color_mode", default="multispectral", help="rgb or multispectral for eurosat")
parser.add_argument("--usecase", default="constrained_hida", help="constrained_hida, u_hida or ss_hida")
parser.add_argument("--idx", type=int, default=-1, help="Index of checkpoint, repetition...")
parser.add_argument("--checkpoint_on", type=int, default=1000, help="After how many steps to save checkpoints, 0 for every epoch")
parser.add_argument("--num_class",type=int, default=8, help="num_class")
parser.add_argument("--contrastive_type", default="all", help="Can be all, source, or none")
parser.add_argument("--constrained", default="random", help="'random' or 'manual' or 'labels' or ''")
parser.add_argument("--processing_type", default="standardization", help="Standarization or normalization")
parser.add_argument("--aug", type=int, default=1, help="Use augmentation (1) or not (0)")
parser.add_argument("--num_constraints", type=int, default=0, help="Percentage of constrained data")
parser.add_argument("--con_links", default="ml_st_cl_st", help="suffix for csv dataset name showing which links are used for constraints")
parser.add_argument("--source", default="resisc", help="Source domain")
parser.add_argument("--target", default="eurosat", help="Target domain")
parser.add_argument("--measure", default="cl_loss_s", help="Choose checkpoint based on this measure")
parser.add_argument("--threshold", default="100", help="Percentage of labelled target data")
parser.add_argument("--batch_norm", type=int, default=0, help="Target domain")
parser.add_argument("--num_sep_layers", type=int, default=2, help="How many layers are separated")
parser.add_argument("--architecture", default="hida", help="hida or ...")
parser.add_argument("--batch_size", type=int, default=24, help="Batch size (half go to source and half to target)")
parser.add_argument("--batch_size_constr", type=int, default=8, help="Batch size for constrained part (half go to source and half to target)")
parser.add_argument("--exp_name", default="exp_01", help="experiment name")
parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
parser.add_argument("--boundary", type=float, default=40, help="Boundary neighbourhood contrastive loss")
parser.add_argument("--wd_loss_on", default="supervised", help="'supervised' or 'unsupervised'")
parser.add_argument("--last_chckp_only", type=int, default=1, help="Save model only at the end")
parser.add_argument("--with_critic", type=int, default=1, help="True default, False for ablation study")
parser.add_argument("--use_unlabelled",type=int, default=0, help="train domain critic separately")

parser.add_argument("--train_dir", type=int, default=None, help="Which train dir to use, None if we use then all")

# parser.add_argument("--csv_file", help="Path of input csv file")
# parser.add_argument("--analyze_return_value", default="acc_t", help="Which values to put in resulting CSVs")
# parser.add_argument("--to_train", type=int, default=1, help="1-Train, 0-Test")
# parser.add_argument("--run_fa_mode", default="plot", help="Whether to calculate feature visualisation or to plot it")
# parser.add_argument("--choose_by", default="last", help="source or last or epoch number")
# parser.add_argument("--dropout", type=int, default=0, help="Use dropout or not")
# parser.add_argument("--save", type=int, default=0, help="to save target images to their classified folders")







args = parser.parse_args()

dataset = args.dataset
num_runs = args.num_runs
color_mode = args.color_mode
usecase = args.usecase
idx = args.idx
checkpoint_on = args.checkpoint_on
num_class = args.num_class
contrastive_type = args.contrastive_type
constrained = args.constrained
processing_type = args.processing_type
aug = args.aug
num_constraints = args.num_constraints
con_links = args.con_links
source = args.source
target = args.target
measure = args.measure
threshold = args.threshold
batch_norm = args.batch_norm
num_sep_layers = args.num_sep_layers
architecture = args.architecture
batch_size = args.batch_size
batch_size_constr = args.batch_size_constr
exp_name = args.exp_name
epochs = args.epochs
boundary = args.boundary
wd_loss_on = args.wd_loss_on
last_chckp_only = args.last_chckp_only
with_critic = args.with_critic
use_unlabelled = args.use_unlabelled

train_dir = args.train_dir


# csv_file = args.csv_file
# analyze_return_value = args.analyze_return_value
# to_train = args.to_train
# run_fa_mode = args.run_fa_mode
# choose_by = args.choose_by
# dropout = args.dropout
# save = args.save



print("dataset:", dataset)
print("number of classes:", num_class)
print("number of runs:", num_runs)
print("usecase:", usecase)
print("color mode:", color_mode)
# print("Use unlabelled", use_unlabelled)
if source:
    print("source", source)
if target:
    print("target", target)
# if batch_norm != None:
#     print("batch norm:", batch_norm)
if exp_name:
    print("experiment name:", exp_name)
# if measure:
#     print("measure:", measure)
if last_chckp_only != None:
    print("last_chckp_only:", last_chckp_only)
# if csv_file != None:
#     print("input csv file", csv_file)
# if threshold != None:
#     print("Threshold", threshold)
if aug:
    print("augmentation:", aug)
# if train_dir:
#     print("train dir:", train_dir)
# if analyze_return_value:
#     print("return value for analyze:", analyze_return_value)
# if to_train:
#     print("to_train:", to_train)
# print("with_critic", with_critic)
print("margin", boundary)
# print("run_fa_mode", run_fa_mode)
print("processing_type", processing_type)
print("batch_size", batch_size)
print("num_sep_layers", num_sep_layers)
print("architecture", architecture)
# print("choose_by", choose_by)
# print("dropout", dropout)
print("checkpoint_on", checkpoint_on)
print("epochs", epochs)
# print("idx", idx)
# print("contrastive_type", contrastive_type)
print("num_constraints", num_constraints)
print("constrained", constrained)
# print("wd_loss_on", wd_loss_on)
# print("save", save)
# print("con_links", con_links)
print("batch_size_constr", batch_size_constr)



















