from pathlib import Path
from PIL import Image
import os

def get_lowest_image_dimensions_from_folder(original_image_folder_directory:str):
    list_of_all_images_height = []
    list_of_all_images_width = []
    for subdir, dirs, files in os.walk(original_image_folder_directory):
        for file in files:
                print(f'Reading file {file}')
                image = Image.open(os.path.join(subdir, file))
                #print(image)
                image_width = int(image.size[0])
                #print(image_height)
                image_height = int(image.size[1])
                #print(image_width)
                aspect_ratio = image_height / image_width
                #print(aspect_ratio)
                list_of_all_images_height.append(image_height)
                list_of_all_images_width.append(image_width)
    min_height = min(list_of_all_images_height)
    min_width = min(list_of_all_images_width)
    print (f'Lowest image width is {min_width} and lowest image height is {min_height}')
            
get_lowest_image_dimensions_from_folder(Path("C:/Users/marko/DS Projects/AirBnB-Data-Modelling/main project file/AirbnbDataSci/images"))  


#### Are the width and height the wrong way round and how to declare correct path for image.save
def resize_images(normalised_height:int, original_image_folder_directory, directory_for_resized_images:str):
    for subdir, dirs, files in os.walk(original_image_folder_directory):
        for file in files:
            f'Reading file {file}'
            image = Image.open(os.path.join(subdir, file))
            print(image)
            image_mode = image.mode
            print(image_mode)
            if image_mode is not 'RGB':
                pass
            else:
                image_width = int(image.size[0])
                print(f'Image width = {image_width}')
                image_height = int(image.size[1])
                print(f'Image height = {image_height}')
                aspect_ratio = image_width / image_height
                print(f'Aspect ratio = {aspect_ratio}')
                new_image_width = int(normalised_height / aspect_ratio) 
                print(new_image_width)  
                image = image.resize((normalised_height, new_image_width), Image.ANTIALIAS)
                file_name = str(f'{directory_for_resized_images}/resized_{file}')
                print(file_name)
                image.save(fp = file_name, format = 'PNG')

if __name__ =='__main__':   
    get_lowest_image_dimensions_from_folder(Path("C:/Users/marko/DS Projects/AirBnB-Data-Modelling/main project file/AirbnbDataSci/images")) 
    resize_images(156, Path("C:/Users/marko/DS Projects/AirBnB-Data-Modelling/main project file/AirbnbDataSci/images"), Path('C:/Users/marko/DS Projects/AirBnB-Data-Modelling/main project file/AirbnbDataSci/processed_images'))
