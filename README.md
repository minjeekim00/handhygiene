# Hand Hygiene Monitoring System




## File structures
### Excel Format
***video_id  video_name	date	target_frame	frame_length***

### CSV Format
*** date	img_name	target	video_id ***
target is 1 for clean(hand hygiene complied)
</br>
target is 0 for not clean
</br></br></br>

## Labeling data

* / your_data_dir
  * / images
</br></br></br>

ex) <pre><code>python add_label.py "your_data_dir" "excel_path" "csv_target_name"</code></pre>
