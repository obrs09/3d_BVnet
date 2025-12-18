import os

import numpy as np

def generate_toy_data(size, length, width, height, outter_value, inner_value, test_value, outter_thickness, test_thickness):
    if test_thickness >= outter_thickness:
        raise ValueError("test_thickness must be less than outter_thickness")
    os.makedirs("toy_data_3d/orig/train", exist_ok=True)
    os.makedirs("toy_data_3d/orig/valid", exist_ok=True)
    os.makedirs("toy_data_3d/orig/test", exist_ok=True)
    os.makedirs("toy_data_3d/orig/train_targets", exist_ok=True)
    os.makedirs("toy_data_3d/orig/valid_targets", exist_ok=True)
    os.makedirs("toy_data_3d/orig/test_targets", exist_ok=True)

    # a = np.full((80, 120, 120, 120, 1), 0.1)
    a = np.full((size, length, width, height, 1), outter_value)
    a[:, outter_thickness:length-outter_thickness, 
      outter_thickness:width-outter_thickness, 
      outter_thickness:height-outter_thickness, :] = inner_value
    num_valid = int(size / 2)
    np.save("toy_data_3d/orig/train/train.npy", a)
    np.save("toy_data_3d/orig/valid/valid.npy", a[:num_valid])
    np.save("toy_data_3d/orig/train.npy", a)
    np.save("toy_data_3d/orig/valid.npy", a[:num_valid])
    del a

    # b = np.full((40, 120, 120, 120, 1), 0.1)
    # b = np.full((40, 120, 120, 120, 1), 0.1)
    b = np.full((num_valid, length, width, height, 1), outter_value)

    # b[:, 20:100, 20:100, 20:100, :] = 0.9
    # b[:, 10:20, 10:110, :] = 0.7
    # b[:, 100:110, 10:110, :] = 0.7
    # b[:, 10:110, 10:20, :] = 0.7
    # b[:, 10:110, 100:110, :] = 0.7

    # b[:, 20:100, 20:100, 20:100, :] = 0.9
    # 在内部立方体周围创建一个值为0.7的“外壳”
    # 沿深度(z)轴的两个面
    # b[:, 10:110, 10:110, 10:20, :] = 0.7
    # b[:, 10:110, 10:110, 100:110, :] = 0.7
    # 沿高度(y)轴的两个面
    # b[:, 10:110, 10:20, 10:110, :] = 0.7
    # b[:, 10:110, 100:110, 10:110, :] = 0.7
    # 沿宽度(x)轴的两个面
    # b[:, 10:20, 10:110, 10:110, :] = 0.7
    # b[:, 100:110, 10:110, 10:110, :] = 0.7

    b[:, outter_thickness : (length - outter_thickness), 
         outter_thickness : (width - outter_thickness),
         outter_thickness : (height - outter_thickness), :] = inner_value

    # 0.7
    b[:, test_thickness : (length - test_thickness), 
         test_thickness : (width - test_thickness), 
         test_thickness : outter_thickness, :] = test_value
    
    b[:, test_thickness : (length - test_thickness), 
         test_thickness : (width - test_thickness), 
         (height - outter_thickness) : (height - test_thickness), :] = test_value


    b[:, test_thickness : (length - test_thickness), 
         test_thickness : outter_thickness,
         test_thickness : (height - test_thickness), :] = test_value
    
    b[:, test_thickness : (length - test_thickness),
         (width - outter_thickness) : (width - test_thickness),
         test_thickness : (height - test_thickness), :] = test_value
    

    b[:, test_thickness : outter_thickness,
         test_thickness : (width - test_thickness),
         test_thickness : (height - test_thickness), :] = test_value
    
    b[:, (length - outter_thickness) : (length - test_thickness),
         test_thickness : (width - test_thickness),
         test_thickness : (height - test_thickness), :] = test_value

    # print('outter_thickness', outter_thickness)
    # print('test_thickness', test_thickness)
    # print('num_valid', num_valid)
    # print('edge thickness', (height - outter_thickness) , (height - test_thickness))

    np.save("toy_data_3d/orig/test/test.npy", b)
    np.save("toy_data_3d/orig/test.npy", b)
    del b

    # c = np.zeros((80, 120, 120, 120, 1))
    # c[:, 20:100, 20:100, 20:100, :] = 1.
    c = np.zeros((size, length, width, height, 1))
    c[:, outter_thickness : (length - outter_thickness),
         outter_thickness : (width - outter_thickness), 
         outter_thickness : (height - outter_thickness), :] = 1.0
    np.save("toy_data_3d/orig/train_targets/train_targets.npy", c)
    np.save("toy_data_3d/orig/valid_targets/valid_targets.npy", c[:num_valid])
    np.save("toy_data_3d/orig/train_targets.npy", c)
    np.save("toy_data_3d/orig/valid_targets.npy", c[:num_valid])

    # c[:, 10:20, 10:110, :] = 1.
    # c[:, 100:110, 10:110, :] = 1.
    # c[:, 10:110, 10:20, :] = 1.
    # c[:, 10:110, 100:110, :] = 1.
    # c[:, 10:110, 10:20, :] = 1.
    # c[:, 10:110, 100:110, :] = 1.

    # c[:, 10:110, 10:110, 10:20, :] = 1.0
    # c[:, 10:110, 10:110, 100:110, :] = 1.0
    # 沿高度(y)轴的两个面
    # c[:, 10:110, 10:20, 10:110, :] = 1.0
    # c[:, 10:110, 100:110, 10:110, :] = 1.0
    # 沿宽度(x)轴的两个面
    # c[:, 10:20, 10:110, 10:110, :] = 1.0
    # c[:, 100:110, 10:110, 10:110, :] = 1.0

    c[:, test_thickness : (length - test_thickness), 
         test_thickness : (width  - test_thickness),
         test_thickness : outter_thickness, :] = 1.0
    c[:, test_thickness : (length - test_thickness),
         test_thickness : (width  - test_thickness),
         (height - outter_thickness) : (height - test_thickness), :] = 1.0
    
    c[:, test_thickness : (length - test_thickness), 
         test_thickness : outter_thickness,
         test_thickness : (height - test_thickness), :] = 1.0
    c[:, test_thickness : (length - test_thickness),
         (width - outter_thickness) : (width - test_thickness),
         test_thickness : (height - test_thickness), :] = 1.0
    
    c[:, test_thickness : outter_thickness,
         test_thickness : (width  - test_thickness),
         test_thickness : (height - test_thickness), :] = 1.0
    c[:, (length - outter_thickness) : (length - test_thickness),
         test_thickness : (width  - test_thickness),
         test_thickness : (height - test_thickness), :] = 1.0

    np.save("toy_data_3d/orig/test_targets/test_targets.npy", c[:num_valid])
    np.save("toy_data_3d/orig/test_targets.npy", c[:num_valid])
    del c

    

if __name__ == "__main__":
    length = 60*2
    width = 60*2
    height = 60*2
    outter_thickness = length//6
    test_thickness = outter_thickness//2
    generate_toy_data(size=40, 
                      length=length, 
                      width=width, 
                      height=height, 
                      outter_value=0.9, 
                      inner_value=0.1, 
                      test_value=0.7, 
                      outter_thickness=outter_thickness, 
                      test_thickness=test_thickness)
