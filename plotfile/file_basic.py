def read_csv2df(path, col_name=None, norm_range=None):
    
    """read csv as dataframe

    Arg:
      col_name: get specific col value
      norm_range: int, norm to [0, int]
    Examples:
     df = read_csv2df(path)
     read_csv2df(path,"Id")
    """

    import pandas as pd

    df = pd.read_csv(path)
    if norm_range:
        df = (df - df.min()) / (df.max() - df.min()) * norm_range
    if col_name:
        return df[col_name]
    else:
        return df


def save_as_csv(save_path, row_data_list, name_row_list=None):
    """write to csv table"""
    """AI is creating summary for save_as_csv
    """
    import csv

    with open(save_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        if name_row_list:
            writer.writerow(name_row_list)
        for i in range(len(row_data_list)):
            row_data = row_data_list[i]
            writer.writerow(row_data)

# more detailed analyses

def save_rows_as_csv(save_path, row_data_list, name_row_list=None):
    """write rows to csv table"""

    import csv

    with open(save_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        if name_row_list:
            writer.writerow(name_row_list)
        for i in range(len(row_data_list)):
            row_data = row_data_list[i]
            writer.writerow(row_data)

def save_cols_as_csv(save_path, col_data_list, name_row_list=None):
    """write cols to csv table"""
    row_data_list = []
    for i in range(len(col_data_list[0])):
        temp_list = []
        for j in range(len(col_data_list)):
            temp_list.append(col_data_list[j][i])
        row_data_list.append(temp_list)
    save_rows_as_csv(save_path, row_data_list, name_row_list)


def rename(old_file,new_name):
    raise NotImplemented

def resize():
    """skiimg import resize to resize numpy array, cv2.resize(img, new_shape)"""
    pass 