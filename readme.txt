14 June 2021
git clone https://github.com/JohannesPausch/Gradient_Sensing_Cell_ML



git pull origin master

do your edits

git add <newfiles>
git commit <changed_files>
git push origin master

... 
doesn't seem to work, but 
git push
seems to work.



Later: 
The editing cycle that seems to work is
git pull origin master


git add <newfiles>
git commit <changed_files>

git push

Later:
I think the failure of "master" has to do with recent changes. It is now
git commit origin main


The cycle is thus
git pull origin master


git add <newfiles>
git commit <changed_files>

git push origin main

