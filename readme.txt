14 June 2021 (pruess)
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




28 June 2021 (pruess)
I have submitted my initial code to simulate the cue particles.

On any Un*x box, you should be able to do
make BrownianParticle
which compiles the code and then run it via
./BrownianParticle
I made 10^6 trajectories the default, so reduce that number to something more sensible via
./BrownianParticle -N 100
which produces 100 trajectories. There are lots of comments in the code.

The output has the format
# PARTICLE 16113 got lost to position -15.4237 -12.6802 1.17781 after 485046 steps at distance 20.0016>20.
# EVENT 2929 16114 2.4283933931160261999 1.6497566833462951053 -0.75460894024879898723 0.65078335842966772429 -0.05149314594962221131 70527
# PARTICLE 16114 got lost to position 11.3682 12.1901 11.0551 after 1550794 steps at distance 20.0012>20.

Any line with “EVENT” is something we want. To filter I do
./BrownianParticle > BrownianParticle_ref.dat
and then
grep EVENT BrownianParticle_ref.dat | sed 's/.*EVENT //' > BrownianParticle_ref.txt
to boil things down to the pure data. I have added that *txt file to GitHub as well.

*** Can someone please analyse that BrownianParticle_ref.txt? We need to make sure that what we are seeing is correct. I would look at the distribution of the x-coordinate at arrival on the sphere and compare to theory. For that we need a normalised histogram of the x-coordinate, which is column 5 of this file.
`

