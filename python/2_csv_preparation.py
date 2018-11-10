import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob



def create_dataframe(img_dir):
    dict_new = {'video_id':[], 'date':[], 'img_name':[]}
    
    for image in tqdm(glob(img_dir+'*.jpg')):
        img_name= image.split('/')[-1].split('.')[0]
        video_id = img_name.split('_')[0]
        date = img_name.split('_')[1]
        
        # create dict for csv
        dict_new['video_id'].append(video_id)
        dict_new['date'].append(date)
        dict_new['img_name'].append(img_name)
        
        # sort by video id and image name
        df = pd.DataFrame(dict_new)
        df.sort_values('img_name', inplace=True)
        df.reset_index(inplace=True)
        df.drop('index', 1, inplace=True)
        
    return df

def label_dataset(df):
    df_excel = pd.read_excel(excel_path)
    dict_new = {'date':[], 'img_name':[], 'video_id':[], 'target':[] }
    temp_array = [] # temp array to save target frame list
    
    for row in tqdm(df.values[:]):
        video_id = int(row[2])
        frame_num = row[1].split('_')[-1][-6:]

        for vid, target, length in df_excel[['video_id', 'target_frame', 'frame_length']].values:
            if video_id is not vid:
                continue
            if target is np.nan:
                continue

            target_frames = target.strip().split(',')
            target_frames = list(map(lambda x: "%.6d" % int(x) ,target_frames))

            if frame_num in target_frames:
                # update target frame list
                temp_array = list(map(lambda n: "%.6d" % (n + int(frame_num)), range(int(length))))
                row = np.append(row, [1])
                continue
            if frame_num in temp_array:
                row = np.append(row, [1])
            else:
                row = np.append(row, [0])

            dict_new['date'].append(row[0]) 
            dict_new['img_name'].append(row[1])
            dict_new['video_id'].append(row[2])
            dict_new['target'].append(row[3])
        
    df_label = pd.DataFrame(dict_new, index=range(len(dict_new['img_name'])))
    return df_label
    
def main():
    parser = argparse.ArgumentParser(description="Labeling hand hygiene images")
    parser.add_argument("data_dir", help="data directory(image dir)")
    parser.add_argument("excel_path", help="excel data path")
    parser.add_argument("target_name", help="csv name to export")

    args = parser.parse_args()

    print("Export csv with label {} from a directory {}".format(args.target_name, args.data_dir))
    
    data_dir = args.data_dir
    img_dir = os.path.join(data_dir, 'images/')
    excel_path = args.excel_path
    label_path = os.path.join(os.getcwd(), args.target_name) 

    # create csv from image name
    df = create_dataframe(img_dir)
    df_label = label_dataset(df)
    
    pd.DataFrame.to_csv(df_label, label_path, index=False)

if __name__ == '__main__':
    main()
                
