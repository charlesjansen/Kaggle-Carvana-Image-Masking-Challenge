# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:22:59 2017

@author: Charles
"""
import glob, re, os

for filename in glob.glob(r'F:\DS-main\Kaggle-main\Carvana Image Masking Challenge\input\New folder'):
     new_name = re.sub("", r'\1_\2\3', filename)
     os.rename(filename, new_name)
     
