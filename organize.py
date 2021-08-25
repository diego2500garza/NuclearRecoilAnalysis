
import os, shutil


list_files = os.listdir()
file_num = input("what number file are we working with?\n")


def rename_file(file_name, file_num):
    prefix = f"Ar{file_num}_"
    new_name = prefix + file_name
    os.rename(file_name, new_name)
    return new_name


def change_dir(file_name, dest):
    shutil.move(file_name, dest)


for file in list_files:
    if (file == "COLLISON.txt"):
        new_name = rename_file(file, file_num)
        change_dir(new_name, 'GAr01 Outputs/Collisions')
    elif (file == "EXYZ.txt"):
        new_name = rename_file(file, file_num)
        change_dir(new_name, 'GAr01 Outputs/EXYZ')
    elif (file == "RANGE_3D.txt"):
        new_name = rename_file(file, file_num)
        change_dir(new_name, 'GAr01 Outputs/Range3D')
    else:
        print(file)
