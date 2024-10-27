#!/bin/bash
#model_test_1data.pyでモデルを検証してその検証結果をcsvに保存する

$day='0825'
$dirname=$day
$num=0
while(1){
    try{
        New-Item ($dirname) -ItemType Directory -ErrorAction Stop
        break
    }
    catch{
        $num=$num+1
        $dirname=($day)+'_'+($num)
    }
}



for($i=0;$i -le 8;$i=$i+1){
    py model_test_1data.py $i >> $dirname/pred$i.csv
}
