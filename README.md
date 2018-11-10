# Hand Hygiene Monitoring System

This is for data preparation of hand hygeiene monitoring project
## Prerequisite
### Prepare your data directory structure like this:
* / your_data_dir
  * / images
  * / bagfiles # your bagfile directory
  * / python
  
## File formats
### Excel Format
***video_id  video_name	date	target_frame	frame_length***
</br>
*frame_length* stands for fps

### CSV Format
***date	img_name	target	video_id***

target is 1 for clean(hand hygiene complied)
</br>
target is 0 for not clean
</br></br>

## To Use
1. Extract rgb images from bag file
<pre><code>python python/1_extract_image_from_bag.py --bag_dir="your_bag_dir" --img_dir="your_image_destination" --data_num="data_number_for_data_addition"</code></pre>
2. Label dataset / export csv file
<pre><code>python python/2_csv_preparation.py  "your_data_dir" "excel_path" "csv_target_name"</code></pre>
3. Split train/valid/test set
<pre><code>python python/3_split_data_for_training.py </code></pre>


