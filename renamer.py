import os

name = ['rin', 'saber', 'sakura']
folder_name = 'pred'

for n in name:
    folder = f"{folder_name}/{n}"

    # Iterate
    for i, file in enumerate(os.listdir(folder)):

        oldName = os.path.join(folder, file)
        # n = os.path.splitext(file)[0]

        b = f'{n}_{i}.png'
        newName = os.path.join(folder, b)

        # Rename the file
        os.rename(oldName, newName)

    res = os.listdir(folder)
    print(res)
