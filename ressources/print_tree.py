import os

def print_directory_tree(folder_path):
    for root, dirs, files in os.walk(folder_path):
        if '.git' in dirs:
            dirs.remove('.git')
        if '.idea' in dirs:
            dirs.remove('.idea')
        level = root.replace(folder_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")


# Usage:
#print_directory_tree('../')
