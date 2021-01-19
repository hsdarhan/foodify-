import os
from shutil import copy, copytree, rmtree
from collections import defaultdict

# Separate source files based on given reference chart
def separate_files(source, target, reference_file):
    
    classes = defaultdict(list)
    with open(reference_file, 'r') as file:
        paths = [line.strip() for line in file.readlines()]
        for x in paths:
            food_class = x.split('/')
            classes[food_class[0]].append(food_class[1] + '.jpg')

    for food_class in classes.keys():
        if not os.path.exists(os.path.join(target,food_class)):
            os.makedirs(os.path.join(target,food_class))
        for a in classes[food_class]:
            copy(os.path.join(source,food_class,a), os.path.join(target,food_class,a))



# Copy individual classes when not all classes are needed
def copy_repository(source, target, reference):

    if os.path.exists(target):
        rmtree(target)
    os.makedirs(target)
    for food_class in reference :
        copytree(os.path.join(source,food_class), os.path.join(target,food_class))
      