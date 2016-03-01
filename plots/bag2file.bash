#!/bin/bash
#set -x #echo on

for bagfile in *.bag; do
	echo "Extracting:" ${bagfile}
	{
		roslaunch bag2file bag2file.launch \
			bagfile_name:=$(pwd)/${bagfile} \
			directory_name:=$(pwd)/"${bagfile%.*}" \
			frame_name:="/Robot_1/base_link"
		mv "${bagfile%.*}"/tf.txt "${bagfile%.*}"/"base".txt
		roslaunch bag2file bag2file.launch \
			bagfile_name:=$(pwd)/${bagfile} \
			directory_name:=$(pwd)/"${bagfile%.*}" \
			frame_name:="/0"
		mv "${bagfile%.*}"/tf.txt "${bagfile%.*}"/"object".txt
	} &> /dev/null
done
