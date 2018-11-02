# Hand Hygiene Monitoring System




## File structures
### Excel Format
#### video_id  video_name	date	target_frame	frame_length

ex) ***"1"	"20180806_1"	"20180806"	"710, 865, 910, 1040, 1100, 1130, 1210, 1370"  	"30"***

target_frame stands for the *starting frame* of hand hygiene action
</br></br>

### CSV Format
#### date	img_name	target	video_id

ex) ***"710"  "20180806" "20180806_1_frame000711"	"1" "1"***

target is 1 for clean(hand hygiene complied)
</br>
target is 0 for not clean
</br></br>

## Labeling data

* / your_data_dir
  * / images
</br></br>

ex) <pre><code>python add_label.py "your_data_dir" "excel_path" "csv_target_name"</code></pre>
