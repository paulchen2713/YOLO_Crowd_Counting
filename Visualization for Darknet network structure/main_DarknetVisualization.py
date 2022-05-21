# -*- coding: utf-8 -*-
"""
Visualization for Darknet (.cfg format)
@author: Tommy Huang, chih.sheng.huang821@gmail.com
"""

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Paul/anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin'

from fun_plot_digraph import plot_graph

path_cfg='yolov3.cfg'
format_output_figure='png'


if __name__=='__main__':
    savefilename=path_cfg.split('.cfg')[0]
    grap_g=plot_graph(path_cfg,savefilename, format=format_output_figure)
    grap_g.view()





