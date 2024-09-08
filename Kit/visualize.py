
import cv2
from pathlib import Path
import natsort

IMG_Path = Path("../labels/test/")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

out_prefix="../label_color/test/"
for k in range(len(IMG_Str)):
    pic = cv2.imread(IMG_Str[k],1)
    pic[pic == 1] = 100
    pic[pic == 2] = 150
    pic[pic == 3] = 200
    pic[pic == 4] = 250
    out_path=out_prefix+Path(IMG_Str[k]).name
    cv2.imwrite(out_path,pic)

