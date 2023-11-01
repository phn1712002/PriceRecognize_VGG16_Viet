@echo off
IF NOT EXIST .git (
    git init
    git remote add origin https://github.com/phn1712002/PriceRecognize_VGG16_Viet.git
)
git stash
git pull origin master
pause