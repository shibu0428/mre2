#変数定義import用ファイル
#bone_mapはボーンの名称を添え字番号で表現している
#parent_pathは親情報を記録している

bone_map = [
    "Hips", #0
    "Spine", #1
    None, #2
    "Chest", #3
    None, #4
    "UpperChest", #5
    None, #6
    "-", #7
    "-", #8
    "Neck", #9
    "Head", #10
    "RightShoulder", #11
    "RightUpperArm", #12
    "RightLowerArm", #13
    "RightHand", #14
    "LeftShoulder", #15
    "LeftUpperArm", #16
    "LeftLowerArm", #17
    "LeftHand", #18
    "RightUpperLeg", #19
    "RightLowerLeg", #20
    "RightFoot", #21
    "RightToes", #22
    "LeftUpperLeg", #23
    "LeftLowerLeg", #24
    "LeftFoot", #25
    "LeftToes" #26
]

parent_path=[
    None,#root
    0,#1.上半身開始
    1,#2
    2,#3
    3,#4
    4,#5
    5,#6
    6,#7
    7,#8
    8,#9
    9,#10
    7,#11
    11,#12
    12,#13
    13,#14
    7,#15
    15,#16
    16,#17
    18,#18
    0,#19
    19,#20
    20,#21
    21,#22
    0,#23
    23,#24
    24,#25
    25#26
]

