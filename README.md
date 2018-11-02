# Hand Hygiene Monitoring System




## Excel Format

### video_id  video_name	date	target_frame	frame_length

ex)
1	20180806_1	20180806	710, 865, 910, 1040, 1100, 1130, 1210, 1370  	30

target_frame stands for the starting point of hand hygiene action


## CSV Format

### date	img_name	target	video_id

ex)
710	  20180806	20180806_1_frame000711	1	  1

target is 1 for clean(hand hygiene complied)
target is 0 for not clean
