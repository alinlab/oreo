#!/bin/bash
for i in  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber \
          DemonAttack Enduro Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster \
          MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown
do
        echo $i
        mkdir -p $i
        cd $i
        gsutil -m cp -n -R gs://atari-replay-datasets/dqn/$i/1 .
        cd ..
done