import pandas as pd
import numpy as np
import pickle
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv(r"HPC_Finished_molmo.csv")

print(df.head())
df = df.drop(df.columns[[0]],axis=1)
print(df.head())
cat = str(df["files"][0]).split('/')[2]
print(cat)

print(df.shape)

total_in_overall = 0
cat_col = pd.DataFrame(columns=["categories"],index=None)
for i,row in df.iterrows():
    cat = str(row["files"]).split('/')[2]
    print(cat)
    new_row = {"categories": cat}
    cat_col.loc[len(cat_col)] = [str(cat)]
    if cat in row["output"]:
        total_in_overall +=1


print(cat_col.head())
cat_col = pd.concat([df,cat_col],ignore_index = True,axis=1)
print(cat_col)
cat_col.columns = ["file","output","category"]

print("cat")

categories = cat_col.category.unique()

biological_cat = ['chameleon','beaver','cashew','ant','nest','flea','frog','cabbage','pumpkin','pig','leopard','rhubarb','brownie','cherry','llama','manatee','opossum','bison','crawfish','bread','tomato','moose','jellyfish','elephant','honeycomb','okra','camel','spareribs''pineapple','chicken','cow','peanut','turtle','eggplant','donkey','antelope','cucumber','ferret','blackberry','starfish','horse','squirrel','hedgehog','spinach','thyme','goat','fig','sheep','barnacle','milk','grape','platypus','giraffe''corn','pear','crocodile','lobster','tumbleweed','apple','mosquito','aloe','wasp','spearmint','cauliflower','anteater','jaguar','moth','grasshopper','bean','raccoon','monkey','turnip','walrus','waterspout','boar','raspberry','blueberry','lettuce','snake','bear','gooseberry','granola','woman','bobcat','parsley','hazelnut','leek','tadpole','garlic','weasel','avocado','dolphin','potato','worm','silverfish','cat','dog','melon','celery','wolf','salamander','mantis','cinnamon','sphinx','hare','wheat','bird','alligator','rice','whale','artichoke','hay','bee','cockroach','groundhog','earwig','koala','scorpion','deer','cannabis','tyrannosaurus','caterpillar','dragonfly','child','chestnut','banana','radish','pomegranate','fern','mango','porcupine','zucchini','strawberry','man','kale','warthog','bamboo','iguana','cranberry','kumquat','chipmunk','asparagus','guava','mushroom','beetle','coconut','chinchilla','onion','octopus','winterberry','sloth','pea','papaya','skunk','mistletoe','cactus','nutmeg','carrot','grapefruit','egg','panda','pecan','hippo','zebra','oyster','acorn','butterfly','kangaroo','lizard','macadamia','rat','plum','snail','eggroll','anthill','beet','hamster','ape','walnut','cumin','peach','flower','lime','spider','apricot','broccoli','lion','mouse','bumblebee','pistachio','tree','fish','breadfruit','lemon','stegosaurus','tiger','nectarine','rabbit','gecko','shrimp','scallion','cheetah','otter','parsnip','persimmon','rhino']
bio_total = 0
non_bio_total = 0


bio_in = 0
non_bio_in = 0
bio_count = []
non_bio_count = []
for cat in categories:
    total_in =0
    total2 = 0
    tempdf = cat_col[cat_col["category"] ==  cat]
    for i,row in tempdf.iterrows():
        total2+=1
        if cat in row["output"]:
            total_in +=1
    #print("----------------------------------------------------------------------------")
    #print("cat = " + cat)
    #print("percent in = " + str(total_in / total2))
    #print("total = " + str(total2))
    if cat in biological_cat:
        bio_total += total2
        bio_in += total_in
        bio_count.append(cat)
    else:
        non_bio_total +=total2
        non_bio_in += total_in
        non_bio_count.append(cat)


print("total_rows = " + str(df.shape[0]))
print("total in = " + str(total_in_overall))
print("Total percent = " + str(total_in_overall / df.shape[0]))
print("categories = " + str(categories))

categories.tofile("categories.csv",sep = ",")
print("-------------------------------Biological -------------------------------------")
print("total in = " + str(bio_total))
print("total pass = " + str(bio_in))
print("Total percent = " + str(bio_in / bio_total))
print("-------------------------------Non Biological ---------------------------------")
print("total in = " + str(non_bio_total))
print("total pass = " + str(non_bio_in))
print("Total percent = " + str(non_bio_in / non_bio_total))
print("-------------------------------Z test------------------------------------------")

count = [bio_in,non_bio_in]
obs = [bio_total,non_bio_total]

stat , p = proportions_ztest(count,obs)

print("stat = "+ str(stat))
print("p = "+str(p))

#0 sub 1 basic 2 sup
