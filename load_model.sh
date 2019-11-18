doomsters_bot_20191118.py
#!/bin/bash
MY_HOME="/home/devcom/Downloads/files/doomsters2019-master"

# - Removing model
cd $MY_HOME
if rm model.data-00000-of-00001 model.index model.meta checkpoint
then
	echo "Model Removed!"
fi

echo  -n "Choose model to copy:"
read model

case $model in
01)
	cp $MY_HOME/model/01/* $MY_HOME
	echo "Model 01 loaded";;
02)
	cp $MY_HOME/model/02/* $MY_HOME
	echo "Model 02 loaded";;
03)
	cp $MY_HOME/model/03/* $MY_HOME
	echo "Model 03 loaded";;
04)
	cp $MY_HOME/model/04/* $MY_HOME
	echo "Model 04 loaded";;
05)
	cp $MY_HOME/model/05/* $MY_HOME
	echo "Model 05 loaded";;
06)
	cp $MY_HOME/model/06/* $MY_HOME
	echo "Model 06 loaded";;
07)
	cp $MY_HOME/model/07/* $MY_HOME
	echo "Model 07 loaded";;
esac
