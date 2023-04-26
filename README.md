# FQ-VIT-TEST
所有代码均基于https://github.com/megvii-research/FQ-ViT    
Many thanks to megvii for your contributions to the community  
主要修改（基本没改）如下  
1.导出图片和权重的txt格式和bin格式，方便部署至硬件中


breakpoint at：  
 `if i % 1 == 0:  
 
            print("debug point")`

cmd：
1.python test_quant.py deit_small ../data/ --quant --ptf --lis --quant-method minmax  
2.python model_test.py  
3.python showimg_int.py ../export/qact_input_int8.txt  

