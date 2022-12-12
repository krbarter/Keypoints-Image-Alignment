"""Use attrib -h in cmd on a file to reveal and delete all hidden files in data input"""
import os

class Directory:
    def __init__(self, directory_name):
        self.directory_name = directory_name
        self.img_input = []
    
    def openDirectory(self):
        if os.path.isdir(self.directory_name) == True:
            for x in os.listdir(self.directory_name):
                self.img_input.append(os.path.join(self.directory_name, x))
        else:
            return "Input is not a valid drectory"
        return self.img_input