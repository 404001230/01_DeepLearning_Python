'''
Author: huaweifan 404001230@qq.com
Date: 2022-05-28 17:58:29
LastEditors: huaweifan 404001230@qq.com
LastEditTime: 2022-05-28 20:21:44
FilePath: /01_DeepLearning/02_Code/02_picture_dog.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import tensorflow as tf
import os

def picture_read():
	#1.构造文件名队列
	tf.compat.v1.train.string_input_producer()
	#2.读取和解码
	#3.批处理
	return None


if __name__ == "__main__":
	file_name = os.listdir("../")
	print(file_name)
	# file_list = [os.path.join("../05_数据/01_dog/", file) for file in file_name]
	# print(file_list)
	# picture_read()