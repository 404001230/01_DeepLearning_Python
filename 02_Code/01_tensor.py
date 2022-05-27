'''
Author: huaweifan 404001230@qq.com
Date: 2022-05-24 18:26:48
LastEditors: huaweifan 404001230@qq.com
LastEditTime: 2022-05-27 22:14:52
FilePath: /01_DeepLearning/02_Code/01_tensor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from mimetypes import init
from pickletools import optimize
import sys
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.disable_eager_execution() 

def tensorflow_demo():
	a_t = tf.constant(2)
	b_t = tf.constant(3)
	c_t = a_t+b_t
	print("c_t = ", c_t)

	#查看默认图 1. 
	# default_g = tf.get_default_graph()
	# print("查看默认图方法1：", default_g)
	print("a_t的图属性：", a_t.graph)
	#开启会话
	with tf.compat.v1.Session() as sess:
		c_t_value = sess.run(c_t)  #若没有run可以用.eval()
		print("c_t_value:", c_t_value)
		print("sess的图属性：", sess.graph)
		#Tesorbord
		tf.compat.v1.summary.FileWriter("./03_TensorBoard_Events", graph=sess.graph)
	#自定义图
	new_g = tf.Graph()
	with new_g.as_default():
		a_new = tf.constant(20)
		b_new = tf.constant(30)
		c_new = a_new + b_new
		print("c_new",c_new)
	#自定图开启会话
	with tf.compat.v1.Session(graph=new_g) as new_sess:
		c_new_value = new_sess.run(c_new)
		print("c_new_value:",c_new_value)
		print("new_sess的图属性",new_sess.graph)
	


	return None

def tensor_demo():
	tensor1 = tf.constant(4.0)
	tensor2 = tf.constant([1,2,3,4])
	linear_squares = tf.constant([[4], [6], [9]], dtype=tf.int32)
	print("tensor1:", tensor1)
	print("tensor2:", tensor2)
	print("linear_squares:", linear_squares)
	return None

def variable_demo():
	a = tf.Variable(initial_value=50)
	b = tf.Variable(initial_value=40)
	c = tf.add(a,b)
	init = tf.compat.v1.global_variables_initializer()#变量初始化
	print("a:", a)
	print("b:", b)
	print("c:", c)
	with tf.compat.v1.Session() as sess:
		sess.run(init)#变量初始化开启会话
		a_value, b_value, c_value = sess.run([a,b,c])
		print("a_value:", a_value)
		print("b_value:", b_value)
		print("c_value:", c_value)
	return None	

def linear_regression():
	#1)准备数据
	X = tf.compat.v1.random_normal(shape = [100,1])
	y_true = tf.matmul(X, [[0.8]]) +0.7
	#2)构造模型，用变量定义模型参数
	weight = tf.Variable(initial_value=tf.compat.v1.random_normal(shape =[1,1]))
	bias = tf.Variable(initial_value=tf.compat.v1.random_normal(shape =[1,1]))
	y_predict = tf.matmul(X, weight)+bias
	#3)构造损失函数
	error = tf.reduce_mean(tf.square(y_predict - y_true))
	#4) 优化损失
	optimizor = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(error)
	init = tf.compat.v1.global_variables_initializer()
	saver = tf.compat.v1.train.Saver()
	with tf.compat.v1.Session() as sess:
		sess.run(init)
		print("before----weight:%f, bias:%f, loss:%f" % (weight.eval(), bias.eval(), error.eval()))
		# for i in range(10):
		# 	sess.run(optimizor)
		# 	# saver.save(sess, "../04_Save_Model/linear.ckpt")
		# 	print("第%d次----weight:%f, bias:%f, loss:%f" % (i+1, weight.eval(), bias.eval(), error.eval()))
		if os.path.exists("../04_Save_Model/checkpoit"):
			saver.restore(sess, "../04_Save_Model/linear.ckpt")
		print("after----weight:%f, bias:%f, loss:%f" % (weight.eval(), bias.eval(), error.eval()))


if __name__ == "__main__":
	# linear_regression()
	# variable_demo()
	# tensor_demo()
	# tensorflow_demo()

