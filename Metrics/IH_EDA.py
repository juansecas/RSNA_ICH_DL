#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:42:34 2019

@author: sebastian
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom

INPUT_PATH = "/home/sebastian/RNSA/"
traindf = pd.read_csv(INPUT_PATH + "stage_1_train.csv")

label = traindf.Label.values
traindf = traindf.ID.str.rsplit("_", n=1, expand=True)
traindf.loc[:, "label"] = label
traindf = traindf.rename({0: "id", 1: "subtype"}, axis=1)
subtype_counts = traindf.groupby("subtype").label.value_counts().unstack()
subtype_counts = subtype_counts.loc[:, 1] / traindf.groupby("subtype").size() * 100


multi_target_count = traindf.groupby("id").label.sum()

#fig, ax = plt.subplots(1,2,figsize=(20,5))

# sns.countplot(traindf.label, ax=ax[0], palette="Reds")
# ax[0].set_xlabel("Binary label")
# ax[0].set_title("How often do we observe a positive label?");


# sns.barplot(x=subtype_counts.index, y=subtype_counts.values, ax=ax[1], palette="Set2")
# plt.xticks(rotation=45); 
# ax[1].set_title("How much binary imbalance do we have?")
# ax[1].set_ylabel("% of positive occurences (1)");

pie, ax = plt.subplots(figsize=[10,6])
labels = subtype_counts.index
plt.pie(x=subtype_counts.values, autopct="%.1f%%", explode=[0.05]*6, labels=labels, pctdistance=0.5)
plt.title("Distribution of positive hemorrhage of RSNA data", fontsize=14);
plt.show()
pie.savefig("/home/sebastian/codigos tesis/RSNA/figures/positive_hemorrhage.png")


zeros = (traindf.label == 0).astype(int).sum(axis=0) / (traindf.label).size *100

ones = (traindf.label == 1).astype(int).sum(axis=0) / (traindf.label).size *100

binary = pd.DataFrame({85.5986, 14.4014})

pie, ax = plt.subplots(figsize=[10,6])
labels = ['No Hemorrhage', 'Any Hemorrhage']
plt.pie(x=binary, autopct="%.1f%%", explode=[0.05]*2, labels=labels, pctdistance=0.5)
plt.title("Distribution of images no hemorrhage vs hemorrhage", fontsize=14);
plt.show()
pie.savefig("/home/sebastian/codigos tesis/RSNA/figures/binary_hemorrhage.png")

# ax = sns.barplot(x=subtype_counts.index, y=subtype_counts.values, palette="Set2")
# ax.set_title("Distribution of positive hemorrhage of RSNA data")
# ax.set_ylabel("Percentage of positive occurences")
# plt.grid()
# plt.show()

# ax2 = sns.countplot(traindf.label, palette="Reds")
# ax2.set_title("Distribution of images no hemorrhage vs hemorrhage")
# ax2.set_ylabel("Total of images")
# ax2.set_xlabel("Binary label")
# #plt.grid()
# plt.show()